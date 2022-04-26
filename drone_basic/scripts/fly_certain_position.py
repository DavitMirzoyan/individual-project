#!/usr/bin/python3

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
import sys
import os

from tf.transformations import euler_from_quaternion
from math import atan2, sqrt, pi

def position_info(data):
    global pub

    task_completed = False

    twist = Twist()

    argv = sys.argv[1:]

    if len(argv) > 3 or len(argv) < 2:
        print("Wrong number of parameters")
        rospy.signal_shutdown("Error")

    try:
        x_position = float(argv[0])
        y_position = float(argv[1])
    except ValueError:
        print("Input value is not number")
        rospy.signal_shutdown("Error")

    drone_position = data.pose[3].position
    drone_orientation = data.pose[3].orientation
    (roll, pitch, drone_direction) = euler_from_quaternion([drone_orientation.x, drone_orientation.y, drone_orientation.z, drone_orientation.w])

    required_angle = atan2(y_position - drone_position.y, x_position - drone_position.x)

    correct_direction = (required_angle * 180/pi) - (drone_direction * 180/pi)
    print(correct_direction, "direction")
    
    
    if (correct_direction > 1 and correct_direction < 180) or correct_direction < -180:
        twist.angular.z = 0.2
        twist.linear.x = 0.0
    elif (correct_direction < -1 and correct_direction > -180) or correct_direction > 180:
        twist.angular.z = -0.2
        twist.linear.x = 0.0

    else:
        distance = sqrt((x_position - drone_position.x)**2 + (y_position - drone_position.y)**2)

        if distance > 0.07:
            twist.angular.z = 0.0
            twist.linear.x = 0.2
        
        else:
            twist.angular.z = 0.0
            twist.linear.x = 0.0
            task_completed = True

    pub.publish(twist)

    if task_completed:
        if len(argv) == 3:
            required_height = float(argv[2])
            os.system("rosrun drone_basic fly_drone.py "+ str(required_height))

        print("drone's x position: ", drone_position.x)
        print("drone's y position: ", drone_position.y)
        print("Position Reached")
        rospy.signal_shutdown("Task Completed")

def listener():
    global pub
    
    rospy.init_node('gazebo', anonymous=True)
    
    rospy.Subscriber('gazebo/model_states', ModelStates, position_info)

    pub = rospy.Publisher('/drone/cmd_vel', Twist, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    listener()
