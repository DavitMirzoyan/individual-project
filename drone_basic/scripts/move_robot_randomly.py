#!/usr/bin/env python3

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

from random import randint

def movebase_client():
    client = actionlib.SimpleActionClient('move_base_robot',MoveBaseAction)
    client.wait_for_server()

    list_of_positions = [[1,0.1], [2, 1], [-6, -1.9], [-8, 0.08], [-7.88, 4.99], 
                         [-6.22, 7.74], [3.58, 5.2], [8, 3], [5, -0.98], [7.98, -1.57]]

    previous_index = -1

    while True:
        random_index = randint(0, 9)

        while random_index == previous_index:
            print("repeat")
            random_index = randint(0, 9)
        
        previous_index = random_index
        random_x_y = list_of_positions[random_index]
        
        x = random_x_y[0]
        y = random_x_y[1]

        position = MoveBaseGoal()
        position.target_pose.header.frame_id = "map"
        position.target_pose.header.stamp = rospy.Time.now()
        position.target_pose.pose.position.x = x
        position.target_pose.pose.position.y = y
        position.target_pose.pose.position.z = 0

        position.target_pose.pose.orientation.w = 0.1

        client.send_goal(position)
        wait = client.wait_for_result()
        if not wait:
            rospy.logerr("Server DOWN ;/ ")

if __name__ == '__main__':
    try:
        rospy.init_node('random_movements')
        movebase_client()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation DONE ")