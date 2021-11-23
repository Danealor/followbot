#!/usr/bin/env python3

from time import process_time
from types import DynamicClassAttribute

from numpy.random.mtrand import randint
import rospy
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CameraInfo
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
    'sat_std': -140,
    'val_offs': 78,
    'val_std': -60,
}

WHITE_FILTER = {
    'hue_mean': 0,
    'hue_std': np.inf,
    'sat_offs': 0,
    'sat_std': 20,
    'val_offs': 78,
    'val_std': -60,
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

# -- Angle Utils -- #


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

def decelerate(dist, stopping_dist, max_speed):
    "Calculates the decelration necessary to stop in a certain distance from a maximum velocity."
    ratio = dist / stopping_dist
    speed = np.sqrt(np.abs(dist / stopping_dist)) * max_speed
    return np.sign(ratio) * min(speed, max_speed)

# -- Coordinate Transformation -- #

WIDTH = 640
HEIGHT = 480


def idx_to_xy(idx):
    return idx[..., ::-1] * (1, -1) + (0, HEIGHT)


def xy_to_idx(xy):
    return ((xy - (0, HEIGHT)) * (1, -1))[..., ::-1]


def xy_to_phy(xy, shape):
    height, width = shape[:2]
    return ((xy - (width/2, height)) * (1/RESOLUTION_X_REAL, -1/RESOLUTION_Y_REAL))


def phy_to_xy(phy, shape):
    height, width = shape[:2]
    return phy * (RESOLUTION_X_REAL, -RESOLUTION_Y_REAL) + (width/2, height)


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

# -- Math Utils -- #

def generate_arc(radius, dist=500., flip=False, num_points=100):
    dists = np.linspace(0, dist, num_points)
    angles = dists / radius
    pts = radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    if flip:
        pts[:,0] *= -1
    return pts

# https://stackoverflow.com/a/51240898
# Given a line with coordinates 'start' and 'end' and the
# coordinates of a point 'pnt' the proc returns the shortest 
# distance from pnt to the line and the coordinates of the 
# nearest point on the line.
#
# 1  Convert the line segment to a vector ('line_vec').
# 2  Create a vector connecting start to pnt ('pnt_vec').
# 3  Find the length of the line vector ('line_len').
# 4  Convert line_vec to a unit vector ('line_unitvec').
# 5  Scale pnt_vec by line_len ('pnt_vec_scaled').
# 6  Get the dot product of line_unitvec and pnt_vec_scaled ('t').
# 7  Ensure t is in the range 0 to 1.
# 8  Use t to get the nearest location on the line to the end
#    of vector pnt_vec_scaled ('nearest').
# 9  Calculate the distance from nearest to pnt_vec_scaled.
# 10 Translate nearest back to the start/end line. 
# Malcolm Kesson 16 Dec 2012
# Edited by Amit 21 Nov 2021
def pnt2line(pnt, start, end):
    line_vec = end - start
    pnt_vec = pnt - start
    line_len = np.linalg.norm(line_vec,axis=-1)
    line_unitvec = line_vec / line_len[...,None]
    pnt_vec_scaled = pnt_vec / line_len[...,None]
    t = np.einsum('...i,...i', line_unitvec, pnt_vec_scaled)
    t = np.clip(t, 0.0, 1.0)
    nearest = line_vec * t[...,None]
    dist = np.linalg.norm(nearest - pnt_vec,axis=-1)
    nearest += start
    return (dist, nearest)

def filter_lines(vertices, point, max_dist):
    dist, nearest = pnt2line(point, vertices[...,0,:], vertices[...,1,:])
    return dist < max_dist

def mask_from_top(mask):
    mask_extended = np.concatenate([mask, np.zeros((1,mask.shape[1]))], axis=0)
    dist = np.argmin(mask_extended, axis=0)
    idx = np.indices(mask.shape)[0]
    return idx < dist

def mask_from_edges(mask):
    mask_top = mask_from_top(mask)
    mask_bot = mask_from_top(mask[::-1])[::-1]

    mask_right = mask_from_top(mask.T).T
    mask_left = mask_from_top(mask[::-1].T).T[::-1]

    return mask_top | mask_bot | mask_right | mask_left

# -- Computer Vision -- #

project_hom = np.array(
    [[1,0,0],
     [0,1,0],
     [0,0,0],
     [0,0,1]])
def homogeneous_cam2gnd(R_euler, T_vec, K_inv):
    rotation = Rotation.from_euler('xyz', R_euler)
    R = rotation.as_matrix()
    R_inv = R.T
    T_vec_inv = -R_inv @ T_vec
    M_inv = np.concatenate([R_inv, T_vec_inv[:,None]], axis=-1)
    H_inv = M_inv @ project_hom @ K_inv
    return H_inv

def homogeneous_translation(t_vec):
    T = np.eye(len(t_vec)+1)
    T[:len(t_vec),-1] = t_vec
    return T

HEIGHT_CAMERA = 0.1
RESOLUTION_X = 15
RESOLUTION_Y = 80

# Hack since the numbers don't quite seem to make sense
RESOLUTION_X_REAL = 60
RESOLUTION_Y_REAL = 50

class PerspectiveShift:
    R = [np.pi/2-0.1,0,0]
    T = [0,0,HEIGHT_CAMERA]
    scale_matrix = np.eye(3) * (RESOLUTION_X, RESOLUTION_Y, 1)

    def __init__(self, camera_matrix):
        self.K = camera_matrix
        self.K_inv = np.linalg.inv(self.K)
        
        H = homogeneous_cam2gnd(self.R, self.T, self.K_inv)
        H_scaled = self.scale_matrix @ H

        self.H = H_scaled

    def warp(self, img, out_size, flip_y=True):
        T_center = homogeneous_translation(np.array([out_size[0]/2,0]))
        H_centered = T_center @ self.H

        img_warped = cv2.warpPerspective(img, H_centered, out_size)
        if flip_y:
            return img_warped[::-1,::-1]
        else:
            return img_warped[::-1,:]

# -- Road Finding -- #

def find_road_start(vertices_left, vertices_right, band_height=20):
    if len(vertices_left) == 0 or len(vertices_right) == 0:
        return None, None

    left_max_y = np.max(vertices_left[...,1])
    right_max_y = np.max(vertices_right[...,1])

    left_min_y = left_max_y - band_height
    right_min_y = right_max_y - band_height
    left_mask = (vertices_left[...,1] > left_min_y) & (vertices_left[...,1] < left_max_y)
    right_mask = (vertices_right[...,1] > right_min_y) & (vertices_right[...,1] < right_max_y)

    if not np.any(left_mask):
        return None, None

    left_corner_idx = np.argmax(vertices_left[left_mask,0])
    left_corner = vertices_left[left_mask][left_corner_idx]

    right_mask &= vertices_right[...,0] > left_corner[0]

    if not np.any(right_mask):
        return None, None

    right_corner_idx = np.argmin(vertices_right[right_mask,0])
    right_corner = vertices_right[right_mask][right_corner_idx]

    return left_corner, right_corner

def extrapolate_vector(vertices_left, vertices_right):
    diff_left = vertices_left[:,1] - vertices_left[:,0]
    diff_right = vertices_right[:,1] - vertices_right[:,0]

    len_left = np.linalg.norm(diff_left, axis=-1)
    len_right = np.linalg.norm(diff_right, axis=-1)

    angles_left = np.pi / 2 - np.arctan(diff_left[:,0] / diff_left[:,1])
    angles_right = np.pi / 2 - np.arctan(diff_right[:,0] / diff_right[:,1])

    theta_left = np.average(angles_left, weights=len_left)
    theta_right = np.average(angles_right, weights=len_right)

    theta = np.mean([theta_left, theta_right])
    if theta > 0:
        theta = theta - np.pi

    length = np.min([np.mean(len_left), np.mean(len_right)])
    return length * np.array([np.cos(theta), np.sin(theta)])

def fit_road(lines_left, lines_right, road_width=50):
    mark_left = np.zeros(lines_left.shape[0], dtype=np.uint8)
    mark_right = np.zeros(lines_right.shape[0], dtype=np.uint8)

    vertices_left = lines_left.reshape(-1,2,2)
    vertices_right = lines_right.reshape(-1,2,2)

    absdiff_left = np.abs(vertices_left[:,1] - vertices_left[:,0])
    absdiff_right = np.abs(vertices_right[:,1] - vertices_right[:,0])

    mask_left_vertical = absdiff_left[:,1] > absdiff_left[:,0]
    mask_right_vertical = absdiff_right[:,1] > absdiff_right[:,0]

    vertices_left_vertical = vertices_left[mask_left_vertical]
    vertices_right_vertical = vertices_right[mask_right_vertical]

    left_corner, right_corner = find_road_start(vertices_left_vertical, vertices_right_vertical)
    if left_corner is None or right_corner is None:
        return None, None, None

    mask_left = filter_lines(vertices_left_vertical, left_corner, road_width)
    mask_right = filter_lines(vertices_right_vertical, right_corner, road_width)

    vertices_road_left = vertices_left_vertical[mask_left]
    vertices_road_right = vertices_right_vertical[mask_right]

    vector = extrapolate_vector(vertices_road_left, vertices_road_right)
    start = (left_corner + right_corner) / 2
    end = start + vector

    midline = np.stack([start, end])

    # Mark our lines
    mark_left[mask_left_vertical] = 1
    mark_right[mask_right_vertical] = 1

    idx_left = np.arange(len(mark_left))[mask_left_vertical][mask_left]
    idx_right = np.arange(len(mark_right))[mask_right_vertical][mask_right]

    mark_left[idx_left] = 2
    mark_right[idx_right] = 2

    return midline, mark_left, mark_right

# -- Odometry Parsing -- #

def parse_odom(msg):
    "Parses a nav_msgs.Odometry message to position and angles"
    pose = msg.pose.pose

    pos = pose.position
    pos_np = np.array([pos.x, pos.y, pos.z])

    quat = pose.orientation
    rotation = Rotation.from_quat([quat.x, quat.y, quat.z, quat.w])
    angles = rotation.as_euler('xyz')

    return pos_np, angles

# -- Control Logic -- #

DIST_LOOKAHEAD = 2.0 # m
DIST_LOOKAHEAD_INT = 2.6 # m
#PIXELS_PER_METER = 380
PIXELS_PER_METER = 300
DIST_CENTERED = 0.25
SPEED_HIGH = 0.5
SPEED_NORMAL = 0.2
SPEED_LOW = 0.075
UPDATE_RATE = 20 # Hz
MIN_LINES = 5
TURN_LOOK_SPEED = 0.25 # rad/s
STRAIGHTAWAY_MIN_TURN_RADIUS = 5 # m
LOSS_MIN = 0.3
LOSS_MAX = 0.6
DIST_MIDLINE_INTERCEPT = 1.5 # m

MAX_VELOCITY_LINEAR = 0.25 # m/s
STOPPING_DISTANCE_LINEAR = 1.0 # m
"Distance to stop moving forward from max speed"

MAX_VELOCITY_ANGULAR = 0.25 # rad/s
STOPPING_DISTANCE_ANGULAR = 0.35 # rad
"Distance to stop turning from max speed"

COLORS_LINES_LEFT = [
    (50, 40, 40), # All lines
    (127, 127, 127), # Vertical lines
    (255, 255, 255)  # Selected vertical lines
]

COLORS_LINES_RIGHT = [
    (0, 64, 64), # All lines
    (0, 150, 150), # Vertical lines
    (64, 255, 255)  # Selected vertical lines
]

class Follower:
    def __init__(self, no_drive=False):
        self.bridge = CvBridge()
        self.filter_yellow = Filter(**YELLOW_FILTER)
        self.filter_white = Filter(**WHITE_FILTER)
        self.rate = rospy.Rate(UPDATE_RATE)
        self.commands = []
        self.img_msg = None
        self.camera_initialized = False
        self.pose_initialized = False
        self.perspective = None
        self.state_reverse = False
        self.running = False
        self.goal_dist = 0.
        self.goal_turn = 0.
        self.distance_traveled = 0.
        self.no_drive = no_drive
        self.moving = False
        self.turning = False
        self.wait_to_stop = False

    def register(self):
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', 
                                          Image, self.image_callback, queue_size=1)
        self.caminfo_sub = rospy.Subscriber('/camera/rgb/camera_info', 
                                          CameraInfo, self.caminfo_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', 
                                          Odometry, self.odom_callback, queue_size=1)
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel',
                                           Twist, queue_size=1)
        rospy.on_shutdown(self.stop)

    def loop(self):
        self.running = True
        self.state = 'main'
        while not rospy.is_shutdown():
            self.branch_state()
            if self.running:
                twist = Twist()
                self.exec_drive(twist)
                self.exec_turn(twist)
                if not self.no_drive:
                    self.cmd_vel_pub.publish(twist)
            try:
                self.rate.sleep()
            except rospy.exceptions.ROSTimeMovedBackwardsException:
                # We pressed Ctrl+R in Gazebo
                self.rate = rospy.Rate(UPDATE_RATE)
                self.rate.sleep()

    def stop(self):
        self.running = False
        self.cmd_vel_pub.publish(Twist())

    ### State Machine ###

    def branch_state(self):
        if self.state == 'main':
            self.state_main()
        elif self.state == 'commands':
            self.state_commands()

    def state_main(self):
        if self.img_msg is not None:
            self.process_image(self.img_msg)

    def state_commands(self):
        if self.wait_to_stop:
            if self.moving or self.turning:
                return
        
        if len(self.commands) == 0:
            print("RETURNING TO MAIN STATE")
            self.state = 'main'
            return

        drive_command, turn_command = self.commands.pop(0)
        self.drive(drive_command)
        self.turn(turn_command)
        self.wait_to_stop = True

    ### Setup goals to be executed in the main loop ###
    def drive(self, distance, reverse=False):
        if not self.pose_initialized:
            return # Not initialized!

        self.state_reverse = reverse
        self.goal_dist = self.distance_traveled + distance

        if reverse:
            print(f"GOAL: Reversing {distance:.2f}m.")
        else:
            print(f"GOAL: Driving {distance:.2f}m.")

    def turn(self, angle):
        if not self.pose_initialized:
            return # Not initialized!

        self.goal_turn = (self.yaw + angle) % (2 * np.pi)
        print(f"GOAL: Turning {np.rad2deg(angle):.0f}deg to {np.rad2deg(self.goal_turn):.0f}deg.")

    ### Execute on our goals through a Twist message ###
    def exec_drive(self, twist):
        if not self.pose_initialized:
            return # Not initialized!

        if self.goal_dist > self.distance_traveled:
            # We want to go forward
            distance_left = self.goal_dist - self.distance_traveled
            twist.linear.x = decelerate(distance_left, STOPPING_DISTANCE_LINEAR, MAX_VELOCITY_LINEAR)
            if self.state_reverse:
                twist.linear.x = -twist.linear.x
            self.moving = np.abs(twist.linear.x) > MAX_VELOCITY_LINEAR / 10

    def exec_turn(self, twist):
        if not self.pose_initialized:
            return # Not initialized!

        angle_left = angle_diff(self.goal_turn, self.yaw)
        twist.angular.z = decelerate(angle_left, STOPPING_DISTANCE_ANGULAR, MAX_VELOCITY_ANGULAR)
        self.turning = np.abs(twist.angular.z) > MAX_VELOCITY_ANGULAR / 10

    ### Receive and parse callbacks from turtlebot ###
    def odom_callback(self, msg):
        if not self.running:
            return

        pos, rot = parse_odom(msg)
        if self.pose_initialized:
            self.distance_traveled += np.linalg.norm(pos - self.pos)
        self.pos = pos
        self.yaw = rot[2]
        self.pose_initialized = True
        
    def image_callback(self, msg):
        self.img_msg = msg

    def caminfo_callback(self, msg):
        if self.camera_initialized:
            return

        camera_matrix = np.array(msg.K).reshape((3,3))
        self.perspective = PerspectiveShift(camera_matrix)
        
        self.camera_initialized = True
        self.caminfo_sub.unregister()

    ### Image Processing and Control ###

    def process_image(self, msg):
        if not self.camera_initialized:
            return

        twist = Twist()

        # Pass our image through some conversions
        width = 400
        height = 400

        img = self.bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        img_projected = self.perspective.warp(img, (width,height))
        img_projected = np.ascontiguousarray(img_projected, dtype=np.uint8)
        hsv = cv2.cvtColor(img_projected, cv2.COLOR_BGR2HSV)
           
        # Filter for road markings
        filter_yellow_img = (self.filter_yellow.apply(hsv) * 255).astype(np.uint8)
        filter_white_img = (self.filter_white.apply(hsv) * 255).astype(np.uint8)

        _, thresh_yellow = cv2.threshold(filter_yellow_img, 40, 255, type=cv2.THRESH_BINARY)
        _, thresh_white = cv2.threshold(filter_white_img, 80, 255, type=cv2.THRESH_BINARY)

        # Mask out sky
        mask_white = thresh_white > 127
        mask_sky = mask_from_edges(mask_white)
        thresh_white[mask_sky] = 0

        # Find lines
        lines_yellow = cv2.HoughLinesP(thresh_yellow, 1, np.pi/180, 40, minLineLength=40, maxLineGap=10)
        lines_white = cv2.HoughLinesP(thresh_white, 1, np.pi/180, 40, minLineLength=40, maxLineGap=10)

        # Check if we have enough lines
        if lines_white is None or lines_white.shape[0] < MIN_LINES:
            # Left side not found! Slow and turn left to find it.
            self.drive(0.1)
            self.turn(np.deg2rad(10))

            print("No left side found, turning left to find it.")
            cv2.imshow("Camera", img)
            cv2.imshow("Overhead", img_projected)
            cv2.waitKey(3)
            return

        if lines_yellow is None or lines_yellow.shape[0] < MIN_LINES:
            # Right side not found! Slow and turn right to find it.
            self.drive(0.1)
            self.turn(np.deg2rad(-10))

            print("No right side found, turning right to find it.")
            cv2.imshow("Camera", img)
            cv2.imshow("Overhead", img_projected)
            cv2.waitKey(3)
            return

        # Find road edges and midline
        lines_white = lines_white[:,0]
        lines_yellow = lines_yellow[:,0]
        midline, mark_left, mark_right = fit_road(lines_white, lines_yellow)

        if midline is None:
            # No road found!
            self.drive(0.25)

            print("No road found!")
            cv2.imshow("Camera", img)
            cv2.imshow("Overhead", img_projected)
            cv2.waitKey(3)
            return

        # Find intercept
        start, end = midline
        vector = end - start
        xs, ys = start
        x_intercept_bot = xs + (height - ys) * (vector[0] / vector[1])
        intercept_bot = np.array((x_intercept_bot, height))

        # Aim for intercept
        intercept_bot_phy = xy_to_phy(intercept_bot, img_projected.shape)
        end_phy = xy_to_phy(end, img_projected.shape)
        vector_phy = end_phy - intercept_bot_phy
        vector_unit = vector_phy / np.linalg.norm(vector_phy)
        pt_intercept_phy = intercept_bot_phy + vector_unit * DIST_MIDLINE_INTERCEPT
        pt_intercept = phy_to_xy(pt_intercept_phy, img_projected.shape)
        dist_phy = np.linalg.norm(end_phy)

        # Look for end of straightaway
        if dist_phy < DIST_LOOKAHEAD:
            self.find_turn(midline, lines_white, lines_yellow, mark_left, mark_right, img_projected.shape)

        # Construct control command
        control_angle = np.arctan2(pt_intercept_phy[1], pt_intercept_phy[0]) - np.pi/2
        drive_dist = np.min([dist_phy, np.linalg.norm(pt_intercept_phy)])
        
        self.turn(control_angle)
        self.drive(drive_dist)

        # Draw image representation
        img_lines = np.zeros_like(img_projected)

        for line, mark in zip(lines_white, mark_left):
            x1, y1, x2, y2 = line
            color = COLORS_LINES_LEFT[mark]
            cv2.line(img_lines,(x1,y1),(x2,y2),color,1)

        for line, mark in zip(lines_yellow, mark_right):
            x1, y1, x2, y2 = line
            color = COLORS_LINES_RIGHT[mark]
            cv2.line(img_lines,(x1,y1),(x2,y2),color,1)

        # Draw midline
        cv2.line(img_projected,start.astype(int),end.astype(int),(0,255,0),2)
        cv2.line(img_lines,start.astype(int),end.astype(int),(0,255,0),1)

        cv2.line(img_projected,intercept_bot.astype(int),start.astype(int),(0,64,0),2)
        cv2.line(img_lines,intercept_bot.astype(int),start.astype(int),(0,64,0),1)

        # Draw control line
        cv2.line(img_projected,(width//2,height),pt_intercept.astype(int),(255,0,0),2)
        cv2.line(img_lines,(width//2,height),pt_intercept.astype(int),(255,0,0),1)
        
        # Display our images
        cv2.imshow("Camera", img)
        cv2.imshow("Overhead", img_projected)
        cv2.imshow("Lines", img_lines)
        cv2.waitKey(3)

    def find_turn(self, midline, lines_left, lines_right, mark_left, mark_right, phy_shape):
        start, end = midline
        start_phy = xy_to_phy(start, phy_shape)
        end_phy = xy_to_phy(end, phy_shape)
        vector = end_phy - start_phy
        theta = np.arctan2(vector[1], vector[0])

        vertices_left = lines_left.reshape(-1,2,2)
        vertices_right = lines_right.reshape(-1,2,2)

        vertices_left_selected = vertices_left[mark_left == 2]
        vertices_left_horizontal = vertices_left[mark_left == 0]

        vertices_left_selend = np.take_along_axis(vertices_left_selected, 
                                        np.argmin(vertices_left_selected[:,:,1],axis=1)[:,None,None], axis=1)[:,0,:]

        len_left_selected = np.linalg.norm(vertices_left_selected[:,1] - vertices_left_selected[:,0], axis=-1)

        vertex_left_selend = np.average(vertices_left_selend, axis=0, weights=len_left_selected)

        corner_vertices_mask = filter_lines(vertices_left_horizontal, vertex_left_selend, 20)
        if np.any(corner_vertices_mask):
            # It's a corner!
            vertices_corner = vertices_left_horizontal[corner_vertices_mask]
            horiz_mid = np.mean(vertices_corner, axis=(0,1))
            if horiz_mid[0] > end[0]:
                self.execute_corner(end_phy, theta, 'right')
            else:
                self.execute_corner(end_phy, theta, 'left')
        else:
            # It's an intersection, find options
            options = []

            vertices_right_selected = vertices_right[mark_right == 2]

            vertices_left_selected_phy = xy_to_phy(vertices_left_selected, phy_shape)
            vertices_right_selected_phy = xy_to_phy(vertices_right_selected, phy_shape)

            dist_left_selected = np.mean(np.max(np.linalg.norm(vertices_left_selected_phy, axis=-1), axis=1))
            dist_right_selected = np.mean(np.max(np.linalg.norm(vertices_right_selected_phy, axis=-1), axis=1))

            if dist_left_selected < DIST_LOOKAHEAD_INT:
                options.append('left')
            if dist_right_selected < DIST_LOOKAHEAD_INT:
                options.append('right')

            diff_above = vertex_left_selend - vertices_left[mark_left==1]
            diff_above_mask = np.all(diff_above[...,1] > 10, axis=1)
            if np.any(diff_above_mask):
                options.append('forward')
                
                # Check if we actually have open space to the left
                if 'left' in options and 'right' in options:
                    shortest_diff_idx = np.argmin(np.linalg.norm(diff_above[diff_above_mask], axis=-1))
                    shortest_diff = diff_above[diff_above_mask].flat[shortest_diff_idx]

                    midpoint = vertex_left_selend + shortest_diff

                    vertices_wall = vertices_right[mark_right==1]
                    mask_wall_left = np.all(vertices_wall[...,0] < midpoint[0], axis=1)
                    mask_wall_above = np.any(vertices_wall[...,1] > midpoint[1], axis=1)
                    mask_wall_below = np.any(vertices_wall[...,1] < midpoint[1], axis=1)

                    if np.any(mask_wall_left & mask_wall_above & mask_wall_below):
                        # There is actually a wall to the left, it's not an option
                        print("Found wall to left.")
                        options.remove('left')


            self.execute_intersection(end_phy, theta, options)


    def execute_corner(self, point, theta, direction):
        print("-- FOUND CORNER --")
        print("Direction:", direction)

        commands = []
        
        # Drive to corner
        dist = np.linalg.norm(point) * 0.8
        angle = np.arctan2(point[1], point[0]) - np.pi/2
        commands.append((dist, angle))

        # Turn forward
        print(np.rad2deg(angle))
        print(np.rad2deg(theta - np.pi/2))
        commands.append((0, theta - np.pi/2 - angle))

        # Drive forward a little
        commands.append((0.1, 0))

        # Turn into corner
        turn_angle = np.pi / 2 if direction == 'left' else -np.pi / 2
        commands.append((0, turn_angle))

        self.state = 'commands'
        self.commands = commands
        self.wait_to_stop = False
        

    def execute_intersection(self, point, theta, options):
        print("-- FOUND INTERSECTION --")
        print("Options:", options)
        
        direction = options[randint(0, len(options))]
        print("I chose:", direction)

        commands = []
        
        # Drive to intersection
        dist = np.linalg.norm(point) * 0.8
        angle = np.arctan2(point[1], point[0]) - np.pi/2
        commands.append((dist, angle))

        # Turn forward
        print(np.rad2deg(angle))
        print(np.rad2deg(theta - np.pi/2))
        commands.append((0, theta - np.pi/2 - angle))

        if direction == 'forward':
            # Drive forward through the intersection
            commands.append((1.0, 0))
        elif direction == 'left':
            # Make a wide turn left
            commands.append((1.0, np.pi / 2))
        elif direction == 'right':
            # Make a tight turn right
            commands.append((0.1, 0))
            commands.append((0.1, -np.pi / 2))
            commands.append((0.1, 0))

        self.state = 'commands'
        self.commands = commands
        self.wait_to_stop = False

def parse_args():
    kwargs = {}

    if rospy.has_param('~no_drive'):
        rospy.loginfo(f"PARAMS: Driving disabled.")
        kwargs['no_drive'] = rospy.get_param('~no_drive')

    return kwargs

def main():
    rospy.init_node('follower')
    kwargs = parse_args()
    follower = Follower(**kwargs)
    follower.register()
    follower.loop()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        print("follower node terminated")