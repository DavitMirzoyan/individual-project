#!/usr/bin/python3

import rospy
from sensor_msgs.msg import Range
from geometry_msgs.msg import Twist

class make_drone_land():
    def __init__(self):
        rospy.Subscriber('/drone/sonar_height', Range, self.height_info)
        self.pub = rospy.Publisher('/drone/cmd_vel', Twist, queue_size=10)
    
    def height_info(self, data):
        task_completed = False

        twist = Twist()  
        
        if data.range > 0.1706:
            twist.linear.z = -0.5
        else:
            twist.linear.z = 0
            task_completed = True

        self.pub.publish(twist)
        #print(data.range)

        if task_completed:
            print("Landing Completed")
            rospy.signal_shutdown("Task Completed")

def main(args=None):   
    rospy.init_node('sonar_height', anonymous=True)  
    make_drone_land()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Exiting")

if __name__ == '__main__':
    main()