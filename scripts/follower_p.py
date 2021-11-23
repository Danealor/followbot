#!/usr/bin/env python3

from time import process_time
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

# -- Probabilistic Masking (Null Hypothesis test) --
import scipy.stats


def pval(dist, points):
    "Returns the p-value of points on a symmetric distribution."
    cumul = dist.cdf(points)
    return 2*np.minimum(cumul, 1-cumul)


def pval_lower(dist, points):
    "Returns the lower p-value of points on a distribution."
    return dist.cdf(points)


def pval_upper(dist, points):
    "Returns the upper p-value of points on a distribution."
    return dist.sf(points)


YELLOW_FILTER = {
    'hue_mean': 29,
    'hue_std': 5,
    'sat_offs': 0,
    'sat_std': -5,
    'val_offs': 78,
    'val_std': -5,
}

WHITE_FILTER = {
    'hue_mean': 0,
    'hue_std': np.inf,
    'sat_offs': 0,
    'sat_std': 5,
    'val_offs': 78,
    'val_std': -5,
}


class Filter:
    def __init__(self, hue_mean=0, hue_std=5, sat_offs=0, sat_std=5, val_offs=0, val_std=5):
        self.hue_dist = scipy.stats.norm(hue_mean, hue_std)
        self.sat_dist = scipy.stats.expon(sat_offs, np.abs(sat_std))
        self.val_dist = scipy.stats.expon(val_offs, np.abs(val_std))
        self.sat_inv = sat_std < 0
        self.val_inv = val_std < 0

    def apply(self, hsv):
        hue, sat, val = np.moveaxis(hsv.astype(float), -1, 0)
        pval_hue = pval(self.hue_dist, hue)
        pval_sat = pval_upper(self.sat_dist, 255 - sat if self.sat_inv else sat)
        pval_val = pval_upper(self.val_dist, 255 - val if self.val_inv else val)
        return pval_hue * pval_sat * pval_val

# -- Angle Utils --


def angle_dist(a, b):
    "Calculates the acute distance between two angles in degrees."
    dist = (a - b) % (2*np.pi)
    return np.where(dist <= np.pi, dist, 2*np.pi - dist)


def angle_diff(a, b):
    "Calculates the acute difference between two angles in degrees."
    dist = (a - b) % (2*np.pi)
    return np.where(dist <= np.pi, dist, dist - 2*np.pi)


def recenter(angles, mid_angle):
    return np.mod(angles - mid_angle + np.pi, 2*np.pi) + mid_angle - np.pi


def search_by_angle(diff, angle):
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    return np.argmin(angle_dist(angles, angle))

# -- Command Control -- #

def smooth_exp(val_old, val_new, alpha):
    return (1 - alpha) * val_old + alpha * val_new

def smooth_exp_cond(val_old, val_new, alpha_rising, alpha_falling):
    alpha = alpha_rising if val_new > val_old else alpha_falling
    return smooth_exp(val_old, val_new, alpha)

def smooth_exp_twist(twist_old, twist_new, 
                        alpha_linear_rising, alpha_linear_falling, 
                        alpha_angular_rising, alpha_angular_falling):
    twist = Twist()
    twist.linear.x = smooth_exp_cond(twist_old.linear.x, twist_new.linear.x, alpha_linear_rising, alpha_linear_falling)
    twist.linear.y = smooth_exp_cond(twist_old.linear.y, twist_new.linear.y, alpha_linear_rising, alpha_linear_falling)
    twist.linear.z = smooth_exp_cond(twist_old.linear.z, twist_new.linear.z, alpha_linear_rising, alpha_linear_falling)
    twist.angular.x = smooth_exp_cond(twist_old.angular.x, twist_new.angular.x, alpha_angular_rising, alpha_angular_falling)
    twist.angular.y = smooth_exp_cond(twist_old.angular.y, twist_new.angular.y, alpha_angular_rising, alpha_angular_falling)
    twist.angular.z = smooth_exp_cond(twist_old.angular.z, twist_new.angular.z, alpha_angular_rising, alpha_angular_falling)
    return twist

def calc_confidence(loss, min, max):
    if loss < min:
        return 1.
    if loss > max:
        return 0.
    confidence = (max - loss) / (max - min)
    return confidence

# -- Coordinate Transformation --


WIDTH = 640
HEIGHT = 480


def idx_to_xy(idx):
    return idx[..., ::-1] * (1, -1) + (0, HEIGHT)


def xy_to_idx(xy):
    return ((xy - (0, HEIGHT)) * (1, -1))[..., ::-1]


def cart_to_rad(coords):
    dist = np.linalg.norm(coords, axis=-1)
    angles = np.arctan2(coords[..., 1], coords[..., 0])
    return np.stack([dist, angles], axis=-1)


Y_VANISHING_NEAR = 105
Y_VANISHING_FAR = 225
CENTER = WIDTH / 2

def dist_tf(dist, y):
    coeff = ((y - Y_VANISHING_FAR) / (Y_VANISHING_NEAR - Y_VANISHING_FAR))
    return np.where(y < Y_VANISHING_FAR, np.inf, dist * coeff)

def perspective_tf(xy):
    x, y = xy[...,0], xy[...,1]
    coeff = (y + Y_VANISHING_NEAR) / (Y_VANISHING_FAR + Y_VANISHING_NEAR)
    x_tf = np.where(y > Y_VANISHING_FAR, np.nan, CENTER + (x - CENTER) * coeff)
    y_tf = np.where(y > Y_VANISHING_FAR, np.inf, y * coeff**2 * (HEIGHT / Y_VANISHING_FAR))
    return np.stack([x_tf, y_tf], axis=-1)

def to_cv(idx):
    return np.round(idx[...,::-1]).astype(int).clip(-9999, 9999)

# -- Generation --

def generate_arc(radius, dist=500., flip=False, num_points=100):
    dists = np.linspace(0, dist, num_points)
    angles = dists / radius
    pts = radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    if flip:
        pts[:,0] *= -1
    return pts

# -- Road Finding --

def find_road_start(left_edges_xy, right_edges_xy, band_height=50):
    left_min_y = np.min(left_edges_xy[...,1])
    right_min_y = np.min(right_edges_xy[...,1])

    #min_y = max(left_min_y, right_min_y)
    #max_y = min_y + band_height
    #left_mask = (left_edges_xy[...,1] > min_y) & (left_edges_xy[...,1] < max_y)
    #right_mask = (right_edges_xy[...,1] > min_y) & (right_edges_xy[...,1] < max_y)

    left_max_y = left_min_y + band_height
    right_max_y = right_min_y + band_height
    left_mask = (left_edges_xy[...,1] > left_min_y) & (left_edges_xy[...,1] < left_max_y)
    right_mask = (right_edges_xy[...,1] > right_min_y) & (right_edges_xy[...,1] < right_max_y)

    if not np.any(left_mask) or not np.any(right_mask):
        return (CENTER, 0)

    left_corner_idx = np.argmax(left_edges_xy[left_mask,0])
    right_corner_idx = np.argmin(right_edges_xy[right_mask,0])

    left_corner = left_edges_xy[left_mask][left_corner_idx]
    right_corner = right_edges_xy[right_mask][right_corner_idx]

    mid = (left_corner + right_corner) / 2
    return mid

# -- Road Tracking and Path Optimization --

def arc_record(center, r, pts, flip):
    v = pts - center
    disp = np.linalg.norm(v, axis=-1) - r
    x, y = np.moveaxis(v, -1, 0)
    x = np.where(flip, -x, x)
    theta = np.arctan2(y, x)
    dist = theta * r
    return np.stack([dist, disp])

def fit_road(left_edges_xy, right_edges_xy, middle, num_test=100, bin_length=20, min_bin_pts=5, max_dist=500, draw_img=None):
    curvatures = np.linspace(-0.01, 0.01, num_test)
    centers = np.stack([middle + 1 / curvatures, np.zeros_like(curvatures)], axis=-1)

    r = np.linalg.norm((middle, 0) - centers, axis=-1)
    turn_right = curvatures > 0

    left_dist, left_disp = arc_record(centers[:,None,:], r[:,None], left_edges_xy[None,:,:], flip=turn_right[:,None])
    right_dist, right_disp = arc_record(centers[:,None,:], r[:,None], right_edges_xy[None,:,:], flip=turn_right[:,None])

    # Correct for directionality
    coeff_left = np.where(turn_right, 1, -1)
    coeff_right = np.where(turn_right, -1, 1)

    left_disp *= coeff_left[:, None]
    right_disp *= coeff_right[:, None]

    # Sort by displacement so that we can search by distance to arc
    left_sort_idx = np.argsort(left_disp, axis=-1)
    right_sort_idx = np.argsort(right_disp, axis=-1)

    left_dist_sorted = np.take_along_axis(left_dist, left_sort_idx, axis=-1)
    right_dist_sorted = np.take_along_axis(right_dist, right_sort_idx, axis=-1)

    left_disp_sorted = np.take_along_axis(left_disp, left_sort_idx, axis=-1)
    right_disp_sorted = np.take_along_axis(right_disp, right_sort_idx, axis=-1)

    # Enforce directionality
    left_valid_mask = left_disp_sorted > 0
    right_valid_mask = right_disp_sorted > 0

    num_bins = int(max_dist / bin_length)
    dist_bins = np.linspace(0, max_dist, num_bins)
    left_bins = np.digitize(left_dist_sorted, dist_bins)
    right_bins = np.digitize(right_dist_sorted, dist_bins)
    left_valid_mask &= left_bins < num_bins
    right_valid_mask &= right_bins < num_bins

    # Ravel the bins into a flattened space
    bin_search_offsets = np.arange(0, num_bins * num_test, num_bins)
    left_bins_offset = left_bins + bin_search_offsets[:,None]
    right_bins_offset = right_bins + bin_search_offsets[:,None]

    left_unique, left_min_idx, left_count_flat = np.unique(left_bins_offset[left_valid_mask], return_index=True, return_counts=True)
    right_unique, right_min_idx, right_count_flat = np.unique(right_bins_offset[right_valid_mask], return_index=True, return_counts=True)

    # Unravel the binning results
    left_count = np.zeros((num_test, num_bins), dtype=int)
    right_count = np.zeros((num_test, num_bins), dtype=int)

    left_count[np.unravel_index(left_unique, left_count.shape)] = left_count_flat
    right_count[np.unravel_index(right_unique, right_count.shape)] = right_count_flat

    left_min = np.zeros((num_test, num_bins), dtype=float)
    right_min = np.zeros((num_test, num_bins), dtype=float)

    left_min[np.unravel_index(left_unique, left_count.shape)] = left_disp_sorted[left_valid_mask][left_min_idx]
    right_min[np.unravel_index(right_unique, right_count.shape)] = right_disp_sorted[right_valid_mask][right_min_idx]

    # Calculate loss
    bin_mask = (left_count >= min_bin_pts) & (right_count >= min_bin_pts)
    loss = np.where(left_min > right_min, left_min / right_min, right_min / left_min)

    # Reward for more bins hit
    #first_bin = np.argmax(bin_mask, axis=-1)
    #last_bin = bin_mask.shape[-1] - np.argmax(bin_mask[:,::-1], axis=-1)
    #bin_dist = last_bin - first_bin
    bin_dist = np.count_nonzero(bin_mask, axis=-1)

    loss_tot = np.mean(loss, axis=-1, where=bin_mask) / np.sqrt(bin_dist)

    # Find best arc candidate
    candidate_mask = np.any(bin_mask, axis=-1)
    if not np.any(candidate_mask):
        return None, 0.0
    best_idx_masked = np.argmin(loss_tot[candidate_mask])
    best_idx = np.arange(candidate_mask.shape[0])[candidate_mask][best_idx_masked]

    # Draw result on image
    if draw_img is not None:
        left_dist = np.zeros((num_test, num_bins), dtype=float)
        right_dist = np.zeros((num_test, num_bins), dtype=float)

        left_dist[np.unravel_index(left_unique, left_count.shape)] = left_dist_sorted[left_valid_mask][left_min_idx]
        right_dist[np.unravel_index(right_unique, right_count.shape)] = right_dist_sorted[right_valid_mask][right_min_idx]

        # Draw left
        theta_left = left_dist[best_idx] / r[best_idx]
        vector_left = np.stack([np.cos(theta_left), np.sin(theta_left)], axis=-1)
        coeff = 1
        if turn_right[best_idx]:
            vector_left[:,0] *= -1
            coeff = -1
        arc_xy = centers[best_idx] + vector_left * r[best_idx]
        left_xy = centers[best_idx] + vector_left * (r[best_idx] - coeff * left_min[best_idx])[:,None]
        
        for a,b in zip(arc_xy, left_xy):
            cv2.line(draw_img, to_cv(xy_to_idx(a)), to_cv(xy_to_idx(b)), (0,0,255), 1)

        # Draw right
        theta_right = right_dist[best_idx] / r[best_idx]
        vector_right = np.stack([np.cos(theta_right), np.sin(theta_right)], axis=-1)
        coeff = 1
        if turn_right[best_idx]:
            vector_right[:,0] *= -1
            coeff = -1
        arc_xy = centers[best_idx] + vector_right * r[best_idx]
        right_xy = centers[best_idx] + vector_right * (r[best_idx] + coeff * right_min[best_idx])[:,None]
        
        for a,b in zip(arc_xy, right_xy):
            cv2.line(draw_img, to_cv(xy_to_idx(a)), to_cv(xy_to_idx(b)), (0,255,0), 1)

    return centers[best_idx], loss_tot[best_idx]

# -- Control Logic --

DIST_LOOKAHEAD = 300
#PIXELS_PER_METER = 380
PIXELS_PER_METER = 300
DIST_CENTERED = 0.25
SPEED_HIGH = 0.5
SPEED_NORMAL = 0.2
SPEED_LOW = 0.075
UPDATE_RATE = 20 # Hz
MIN_POINTS = 25
TURN_LOOK_SPEED = 0.25 # rad/s
STRAIGHTAWAY_MIN_TURN_RADIUS = 5 # m
COMMAND_QUEUE_SIZE = 3
ANGLE_RAMP_FACTOR = 0.5
SPEED_RAMP_UP_FACTOR = 0.25
SPEED_RAMP_DOWN_FACTOR = 0.75
LOSS_MIN = 0.3
LOSS_MAX = 0.6

class Follower:
    def __init__(self):
        self.bridge = CvBridge()
        self.filter_yellow = Filter(**YELLOW_FILTER)
        self.filter_white = Filter(**WHITE_FILTER)
        self.rate = rospy.Rate(UPDATE_RATE)
        self.current_command = Twist()
        self.command_queue = []
        self.backup_command = None
        self.img_msg = None
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', 
                                          Image, self.image_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel',
                                           Twist, queue_size=1)

    def loop(self):
        while not rospy.is_shutdown():
            if self.img_msg is not None:
                self.process_image(self.img_msg)
                self.execute_command()
            try:
                self.rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                # We pressed Ctrl+R in Gazebo
                self.rate = rospy.Rate(UPDATE_RATE)
                self.rate.sleep()

    def queue_command(self, twist_command, confidence=1.):
        while len(self.command_queue) < COMMAND_QUEUE_SIZE:
            self.command_queue.insert(0, (twist_command, confidence))

    def execute_command(self):
        twist_command = self.backup_command
        confidence = 1.0
        if len(self.command_queue):
            twist_command, confidence = self.command_queue.pop()
        if twist_command is not None:
            self.current_command = smooth_exp_twist(self.current_command, twist_command,
                alpha_linear_rising=SPEED_RAMP_UP_FACTOR,
                alpha_linear_falling=SPEED_RAMP_DOWN_FACTOR,
                alpha_angular_rising=ANGLE_RAMP_FACTOR * confidence,
                alpha_angular_falling=ANGLE_RAMP_FACTOR * confidence)
            self.cmd_vel_pub.publish(self.current_command)
        

    def image_callback(self, msg):
        self.img_msg = msg

    def process_image(self, msg):
        twist = Twist()

        # Pass our image through some conversions
        img = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Filter for road markings
        filter_yellow_img = (self.filter_yellow.apply(hsv) * 255).astype(np.uint8)
        filter_white_img = (self.filter_white.apply(hsv) * 255).astype(np.uint8)
        filter_white_img[:260,:] = 0 # Erase sky

        # Find road edges
        edges_yellow = cv2.Canny(filter_yellow_img, 20, 240)
        edges_white = cv2.Canny(filter_white_img, 20, 240)

        # Highlight edges in main image
        img += edges_yellow[...,None]# * (0,0,1)
        img += edges_white[...,None]# * (0,1,0)

        # Convert to top-down coordinates
        right_edges_idx = np.moveaxis(np.indices(edges_white.shape),0,-1)[edges_white>127].reshape(-1,2)
        right_edges_xy = perspective_tf(idx_to_xy(right_edges_idx))

        left_edges_idx = np.moveaxis(np.indices(edges_yellow.shape),0,-1)[edges_yellow>127].reshape(-1,2)
        left_edges_xy = perspective_tf(idx_to_xy(left_edges_idx))

        # Draw top-down perspective
        img_overhead = np.zeros_like(img)
        img_overhead[tuple(xy_to_idx(left_edges_xy).astype(int).clip((0,0),(HEIGHT-1,WIDTH-1)).T)] = (0, 255, 255)
        img_overhead[tuple(xy_to_idx(right_edges_xy).astype(int).clip((0,0),(HEIGHT-1,WIDTH-1)).T)] = (255, 255, 255)

        if left_edges_xy.shape[0] < MIN_POINTS:
            # Left side not found! Slow and turn left to find it.
            twist.linear.x = SPEED_LOW
            twist.angular.z = TURN_LOOK_SPEED
            self.backup_command = twist

            print("No left side found, turning left to find it.")
            cv2.imshow("Camera", img)
            cv2.imshow("Overhead", img_overhead)
            cv2.waitKey(3)
            return

        if right_edges_xy.shape[0] < MIN_POINTS:
            # Right side not found! Slow and turn left to find it.
            twist.linear.x = SPEED_LOW
            twist.angular.z = -TURN_LOOK_SPEED
            self.backup_command = twist

            print("No right side found, turning right to find it.")
            cv2.imshow("Camera", img)
            cv2.imshow("Overhead", img_overhead)
            cv2.waitKey(3)
            return

        # Mark our road
        middle = find_road_start(left_edges_xy, right_edges_xy)
        x_mid = middle[0]

        # Check if we're centered on the road
        road_offs = (x_mid - CENTER) / PIXELS_PER_METER
        centered = np.abs(road_offs) < DIST_CENTERED

        # Find our path
        centerpoint, loss = fit_road(left_edges_xy, right_edges_xy, x_mid, max_dist=DIST_LOOKAHEAD, draw_img=img_overhead)
        confidence = calc_confidence(loss, LOSS_MIN, LOSS_MAX)

        if centerpoint is None:
            # No road found!
            twist.linear.x = SPEED_LOW
            self.backup_command = twist

            print("No road found!")
            cv2.imshow("Camera", img)
            cv2.imshow("Overhead", img_overhead)
            cv2.waitKey(3)
            return

        # Infer radius
        radius_signed = x_mid - centerpoint[0]
        radius = np.abs(radius_signed)

        # Draw path on overhead
        arc = generate_arc(radius, flip=radius_signed<0, num_points=10, dist=DIST_LOOKAHEAD) + centerpoint
        arc_cv = to_cv(xy_to_idx(arc).reshape(-1,2))
        last_pt = None
        arc_color = (255, 0, 0) if centered else (255, 255, 255)
        img_overhead_solid = img_overhead.copy()
        for pt in arc_cv:
            if last_pt is not None:
                cv2.line(img_overhead_solid, last_pt, pt, arc_color, 2)
            cv2.circle(img_overhead_solid, pt, 5, arc_color, -1)
            last_pt = pt
        img_overhead = cv2.addWeighted(img_overhead, 1 - confidence, img_overhead_solid, confidence, 0)

        # Convert to control command
        w = 0.
        v = 0.
        if centered and confidence > 0.25:
            turning_radius_signed = radius_signed / PIXELS_PER_METER
            v = SPEED_NORMAL * confidence if np.abs(turning_radius_signed) < STRAIGHTAWAY_MIN_TURN_RADIUS else SPEED_HIGH * confidence
            w = v / turning_radius_signed - road_offs / 5
        else:
            print(f"Turning back to road at {road_offs:.2f} m")
            v = SPEED_NORMAL * confidence
            w = -road_offs / 3
            confidence = 1.0
        
        if not np.isnan(w) and not np.isinf(w):
            twist.angular.z = w
        twist.linear.x = v
        print(f"Turning {np.rad2deg(w):.1f} deg/s")
        print(f"Driving {v:.2f} m/s")
        self.queue_command(twist, confidence)
        
        # Display our images
        cv2.imshow("Camera", img)
        cv2.imshow("Overhead", img_overhead)
        cv2.waitKey(3)

rospy.init_node('follower')
follower = Follower()
follower.loop()
