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
    'sat_std': 5,
    'val_offs': 78,
    'val_std': 5,
}


class Filter:
    def __init__(self, hue_mean=0, hue_std=5, sat_offs=0, sat_std=5, val_offs=0, val_std=5):
        self.hue_dist = scipy.stats.norm(hue_mean, hue_std)
        self.sat_dist = scipy.stats.expon(sat_offs, sat_std)
        self.val_dist = scipy.stats.expon(val_offs, val_std)
        self.inv_tf = lambda x: 255 - x

    def apply(self, hsv):
        hue, sat, val = np.moveaxis(hsv.astype(float), -1, 0)
        pval_hue = pval(self.hue_dist, hue)
        pval_sat = pval_upper(self.sat_dist, self.inv_tf(sat))
        pval_val = pval_upper(self.val_dist, self.inv_tf(val))
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


Y_VANISHING_NEAR = 375
Y_VANISHING_FAR = 250
CENTER = WIDTH / 2


def dist_tf(dist, y):
    coeff = ((y - Y_VANISHING_FAR) / (Y_VANISHING_NEAR - Y_VANISHING_FAR))
    return np.where(y < Y_VANISHING_FAR, np.inf, dist * coeff)


def perspective_tf(xy):
    x, y = xy[..., 0], xy[..., 1]
    coeff = y / (HEIGHT - Y_VANISHING_FAR)
    x_tf = np.where(y > HEIGHT - Y_VANISHING_FAR, np.inf,
                    CENTER + (x - CENTER) * coeff)
    return np.stack([x_tf, y], axis=-1)

def to_cv(idx):
    return np.round(idx[...,::-1]).astype(int).clip(-9999, 9999)

# -- Road Tracking --

def mark_road(img, edges, dist=50, max_iter=10, angle_lim=45, angle_tol=5):
    # Transform points from perspective to top-down
    edge_idx = np.moveaxis(np.indices(edges.shape),
                           0, -1)[edges > 127].reshape(-1, 2)
    edge_xy = perspective_tf(idx_to_xy(edge_idx))

    # Draw top-down perspective
    img_overhead = np.zeros_like(img)
    img_overhead[tuple(xy_to_idx(edge_xy).astype(int).clip((0,0),(HEIGHT-1,WIDTH-1)).T)] = (255, 255, 255)

    # Save midpoints to return as road
    mids = []
    mids_xy = []
    normals = []

    try:
        # Find start closest to bottom of screen (angle-wise)
        center = (CENTER, 0)
        diff_xy = idx_to_xy(edge_idx) - center
        edge_rad = recenter(np.arctan2(diff_xy[:, 1], diff_xy[:, 0]), np.deg2rad(90))
        left_idx, right_idx = np.argmax(edge_rad), np.argmin(edge_rad)
        left_xy, right_xy = edge_xy[[left_idx, right_idx]]
        left, right = edge_idx[[left_idx, right_idx]]

        for _ in range(max_iter):
            # Calculate normal angle
            diff_xy = right_xy - left_xy
            line_rad = np.arctan2(diff_xy[1], diff_xy[0])
            theta = line_rad + np.deg2rad(90)

            # Calculate and save midpoints
            mid = np.mean([left, right], axis=0)
            mid_xy = np.mean([left_xy, right_xy], axis=0)
            mids.append(mid)
            mids_xy.append(mid_xy)

            # Draw "rail" with midpoint
            cv2.line(img, to_cv(left), to_cv(right), (0, 255, 0), 2)
            cv2.circle(img, to_cv(mid), 5, (0, 0, 255), -1)

            cv2.line(img_overhead, to_cv(xy_to_idx(left_xy)), to_cv(xy_to_idx(right_xy)), (0, 255, 0), 2)
            cv2.circle(img_overhead, to_cv(xy_to_idx(mid_xy)), 5, (0, 0, 255), -1)

            # Draw midpoint connection
            if len(mids) > 1:
                prev_mid = mids[-2]
                prev_mid_xy = mids_xy[-2]
                cv2.line(img, to_cv(prev_mid), to_cv(mid), (0,0,255),2)
                cv2.line(img_overhead, to_cv(xy_to_idx(prev_mid_xy)), to_cv(xy_to_idx(mid_xy)), (0,0,255),2)

            # Continue to next point
            normal = np.array([np.cos(theta), np.sin(theta)])
            next_pt_xy = mid_xy + dist * normal
            normals.append(normal)

            # Find new edge points to left and right crossing next_pt
            diff_xy = edge_xy - next_pt_xy
            radial = cart_to_rad(diff_xy)
            left_filter = angle_dist(radial[:,1], theta + np.deg2rad(90)) < np.deg2rad(angle_lim)
            right_idx = np.argmin(angle_dist(radial[:,1], radial[left_filter,1,None] - np.deg2rad(180)), axis=-1)
            opposite_filter = angle_dist(radial[left_filter,1], radial[right_idx,1] - np.deg2rad(180)) < np.deg2rad(angle_tol)

            # Choose next line based on minimum distance
            distances = radial[left_filter,0][opposite_filter] + radial[right_idx,0][opposite_filter]
            line_idx = np.argmin(distances)
            left_xy = edge_xy[left_filter][opposite_filter][line_idx]
            right_xy = edge_xy[right_idx][opposite_filter][line_idx]
            left = edge_idx[left_filter][opposite_filter][line_idx]
            right = edge_idx[right_idx][opposite_filter][line_idx]

            # Save last midpoint
            prev_mid = mid
            prev_mid_xy = mid_xy
    except ValueError as e:
        pass

    return img_overhead, np.array(mids), np.array(mids_xy), np.array(normals)

# -- Path Optimization --

def cross_arc(point, normal, radius):
    x0, y0 = np.moveaxis(point, -1, 0)
    xN, yN = np.moveaxis(normal, -1, 0)

    # Quadratic formula
    a = xN**2 + yN**2
    b = 2*(x0*xN + y0*yN)
    c = x0**2 + y0**2 - radius**2

    # Solve
    discriminant = b**2 - 4*a*c
    valid = discriminant >= 0
    sqrt_disc = np.where(valid, np.sqrt(discriminant,where=valid), np.nan)
    pos = (-b + sqrt_disc) / (2*a)
    neg = (-b - sqrt_disc) / (2*a)
    t = np.stack([pos, neg], axis=-1)

    # Return points
    return point[...,None,:] + t[...,None] * normal[...,None,:]

class PathFinder:
    def __init__(self, dist_drive=100, num_trials=100):
        # Create centerpoint candidates
        self.dist_drive = dist_drive
        self.middle = (CENTER,0)
        search_x = np.linspace(-WIDTH, WIDTH*2, num_trials)
        search_x = search_x[np.abs(search_x - self.middle[0]) > 2 * self.dist_drive / np.pi]
        self.centerpoints = np.stack([search_x,np.zeros_like(search_x)],axis=-1)

        # Precalculate some variables
        self.start_offs = self.middle - self.centerpoints
        self.radius = np.linalg.norm(self.start_offs, axis=-1)
        self.angle_drive = self.dist_drive / self.radius

    def set_middle(self, mid):
        change = mid - self.middle
        self.middle = mid
        self.centerpoints += change

    def find_path(self, pts, normal):
        # Associate each point with point on arc through its given normal
        pts_offs = pts[None,...] - self.centerpoints[:,None,:]
        pts_arc_offs = cross_arc(pts_offs, normal[None,...], self.radius[:,None])

        # Calculate angle on arc and filter to limit by travel distance
        angle = np.arctan2(pts_arc_offs[...,1], pts_arc_offs[...,0])
        angle = np.where(self.start_offs[...,None,None,0] > 0, angle, np.pi - angle)
        filter_arc = (angle >= 0) & (angle < self.angle_drive[:,None,None])

        # Calculate R^2 (alternate, not standard) and reduce down to find best path candidate
        diff = pts_arc_offs - pts_offs[...,None,:]
        dist = np.sum(diff**2,axis=-1)
        R2 = np.min(dist, axis=-1, where=filter_arc, initial=np.inf)
        R2_tot = np.mean(R2, axis=-1, where=~np.isinf(R2))
        best_candidate_idx = np.argmin(R2_tot)

        centerpoint_chosen = self.centerpoints[best_candidate_idx]
        pts_arc_chosen = pts_arc_offs[best_candidate_idx] + centerpoint_chosen
        pts_arc_drive_chosen = pts_arc_chosen[filter_arc[best_candidate_idx]]

        return centerpoint_chosen, pts_arc_drive_chosen

# -- Control Logic --

DIST_LOOKAHEAD = 200
#PIXELS_PER_METER = 380
PIXELS_PER_METER = 150
DIST_CENTERED = 0.5
SPEED = 0.2
SPEED_LOW = 0.1
UPDATE_RATE = 4 # Hz

class Follower:
    def __init__(self):
        self.bridge = CvBridge()
        self.filter_yellow = Filter(**YELLOW_FILTER)
        self.path_finder = PathFinder(dist_drive=DIST_LOOKAHEAD)
        self.rate = rospy.Rate(UPDATE_RATE)
        self.turning_velocity = 0.
        # cv2.namedWindow("window", 1)
        self.img_msg = None
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', 
                                          Image, self.image_callback)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel',
                                           Twist, queue_size=1)

    def loop(self):
        while not rospy.is_shutdown():
            if self.img_msg is not None:
                self.process_image(self.img_msg)
            try:
                self.rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                # We pressed Ctrl+R in Gazebo
                self.rate = rospy.Rate(UPDATE_RATE)
                self.rate.sleep()

    def image_callback(self, msg):
        self.img_msg = msg

    def process_image(self, msg):
        # Pass our image through some conversions
        img = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        filter_img = (self.filter_yellow.apply(hsv) * 255).astype(np.uint8)
        edges = cv2.Canny(filter_img, 20, 240)
        img += edges[...,None]

        #cv2.imshow("Filter", filter_img)

        # Mark our road
        img_overhead, road, road_xy, vectors = mark_road(img, edges, dist=15, max_iter=50)

        if not road.size:
            # No road found!
            twist = Twist()
            twist.linear.x = SPEED
            self.cmd_vel_pub.publish(twist)

            cv2.imshow("Camera", img)
            cv2.waitKey(3)
            return

        # Check if we're centered on the road
        road_mid_xy = road_xy[0]
        road_mid = road[0]
        road_offs = (road_mid_xy - CENTER)[0] / PIXELS_PER_METER
        print(road_offs)
        centered = np.abs(road_offs) < DIST_CENTERED

        # Find our path
        normals = vectors[...,::-1] * (-1,1)
        #self.path_finder.set_middle(np.array((CENTER,road_mid_xy[1])))
        centerpoint, arc = self.path_finder.find_path(road_xy, normals)

        # Draw path on overhead
        arc_cv = to_cv(xy_to_idx(arc).reshape(-1,2))
        last_pt = None
        arc_color = (255, 0, 0) if centered else (255, 255, 255)
        for pt in arc_cv:
            if last_pt is not None:
                cv2.line(img_overhead, last_pt, pt, arc_color, 2)
            cv2.circle(img_overhead, pt, 5, arc_color, -1)
            last_pt = pt

        # Convert to control command
        twist = Twist()
        w = 0.
        if centered:
            turning_radius_signed = (road_mid_xy[0] - centerpoint[0]) / PIXELS_PER_METER
            w = SPEED / turning_radius_signed
            twist.linear.x = SPEED
        else:
            print(f"Turning back to road at {road_offs} m")
            w = -road_offs / 10
            twist.linear.x = SPEED_LOW
        
        if not np.isnan(w) and not np.isinf(w):
            self.turning_velocity = self.turning_velocity * 0.5 + w * 0.5
        print(f"Turning {np.rad2deg(self.turning_velocity)} deg/s")
        twist.angular.z = self.turning_velocity
        self.cmd_vel_pub.publish(twist)
        
        # Display our images
        cv2.imshow("Camera", img)
        cv2.imshow("Overhead", img_overhead)
        cv2.waitKey(3)

rospy.init_node('follower')
follower = Follower()
follower.loop()
