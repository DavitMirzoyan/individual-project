#!/usr/bin/python3

import rospy
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist
import sys

class make_drone_fly():
    def __init__(self, height_value):
        
        self.initial_check = True
        self.go_down = False
        rospy.Subscriber('drone/sonar_height', Range, self.height_info, height_value)
        self.pub = rospy.Publisher('drone/cmd_vel', Twist, queue_size=10)
    
    def height_info(self, data, height_value):
        if height_value > 10:
            height_value = 10

        task_completed = False
        drone_velocity = Twist()

        if data.range > height_value and self.initial_check:
            self.go_down = True
        self.initial_check = False

        if data.range < height_value and not self.go_down:
            drone_velocity.linear.z = 0.35
        elif data.range > height_value and self.go_down:
            drone_velocity.linear.z = -0.35
        else:
            drone_velocity.linear.z = 0
            task_completed = True

        #print(round(data.range,4))
        self.pub.publish(drone_velocity)

        if task_completed:
            print("Flying Completed")
            rospy.signal_shutdown("Task Completed")

def main(args=None):
    argv = sys.argv[1:]

    if len(argv) != 1:
        print("Wrong number of parameters")
        rospy.signal_shutdown("Error")
    
    try:
        height_value = float(argv[0])
    except ValueError:
        print("Input value is not number")
        rospy.signal_shutdown("Error")

    rospy.init_node('sonar_height', anonymous=True)                                           
    make_drone_fly(height_value)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Exiting")

if __name__ == '__main__':
    main()