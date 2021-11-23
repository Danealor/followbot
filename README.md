# Followbot Homework

**Student Name:** Amit Blonder<br/>
**Student ID:** 501 875 93

## Follow a path

1. Configure a `.launch` file to spawn a `waffle_pi` turtlebot in the **`course.world`** world.
    - This world uses the `course.material` material, which in turn points to the `course_6.png` ground image.
2. `roslaunch` the `.launch` file
    - e.g. `roslaunch turtlebot_gazebo turtlebot_course.launch`
3. `rosrun` the **`follower_p.py`** file to activate the followbot.
    - `rosrun followbot follower_p.py`
4. Position the windows such that Gazebo, the Terminal and the "Overhead" camera window are all visible.
5. Track the followbot and take note of how it interprents the world and path in front of it.

## Follow a path

1. Configure a `.launch` file to spawn a `waffle_pi` turtlebot in the **`course_intersections.world`** world.
    - This world uses the `course_intersections.material` material, which in turn points to the `course_intersections_3.png` ground image.
2. `roslaunch` the `.launch` file
    - e.g. `roslaunch turtlebot_gazebo turtlebot_course_intersections.launch`
3. `rosrun` the **`follower_p_intersections.py`** file to activate the followbot.
    - `rosrun followbot follower_p_intersections.py`
4. Position the windows such that Gazebo, the Terminal and the "Overhead" and "Lines" camera windows are all visible.
5. Track the followbot and take note of how it interprents the world and path in front of it.
    - The followbot will announce in the terminal when it encounters a corner or an intersection, and then drive up to them and proceed to execute a set of commands to navigate them. It will choose randomly what turn to take at an intersection, and will announce its choice.

The followbot can also be run in "no_drive" mode, where the drone will not move but continue to analyze the camera imagery.

Turn this option on with a parameter:
```
rosrun followbot follower_p_intersections.py _no_drive:=True
```

In the same Gazebo session, turn this option back off by:
```
rosrun followbot follower_p_intersections.py _no_drive:=False
```