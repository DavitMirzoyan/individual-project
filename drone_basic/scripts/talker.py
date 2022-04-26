#!/usr/bin/python3
# Software License Agreement (BSD License)
#
# Copyright (c) 2008, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Revision $Id$

## Simple talker demo that published std_msgs/Strings messages
## to the 'chatter' topic

# roslaunch drone_basic bring_hector_drone.launch
# rosrun drone_basic talker.py

# <!-- rosrun teleop_twist_keyboard teleop_twist_keyboard.py --remap /cmd_vel:=/drone/cmd_vel  -->
import rospy
from geometry_msgs.msg import Twist

def talker():
    twist = Twist()
    #twist.linear.x = 0.2
    #twist.linear.y = 0.2
    twist.linear.z = 0.3
    twist.linear.z = 1
    
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub.publish(twist)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass

#!/usr/bin/python3

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
import sys
import os

from tf.transformations import euler_from_quaternion
from math import atan2, sqrt, pow

global direction_found
direction_found = False

global prev_distance
prev_distance = 1000

def position_info(data):
    global pub
    global direction_found
    global prev_distance

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

    drone_position = data.pose[1].position
    drone_orientation = data.pose[1].orientation
    (roll, pitch, drone_direction) = euler_from_quaternion([drone_orientation.x, drone_orientation.y, drone_orientation.z, drone_orientation.w])

    required_angle = atan2(y_position - drone_position.y, x_position - drone_position.x)

    correct_direction = required_angle - drone_direction
    #print(correct_direction, "angle")

    
    if correct_direction > 0.1 and not direction_found:
        twist.angular.z = 0.5
        twist.linear.x = 0.0

    elif correct_direction < -0.1 and not direction_found:
        twist.angular.z = -0.5
        twist.linear.x = 0.0

    else:
        print(correct_direction)
        direction_found = True
        distance = sqrt(pow((x_position - drone_position.x),2) + pow((y_position - drone_position.y),2))
        
        print("drone's x position: ", drone_position.x)
        print("drone's y position: ", drone_position.y)
        print(distance)
        print(prev_distance)
        
        if prev_distance >= distance:
            twist.angular.z = 0.0
            twist.linear.x = 0.5
            prev_distance = distance
        
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

    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    rospy.spin()

if __name__ == '__main__':
    listener()

#!/usr/bin/python3

import rospy

import cv2 

from cv_bridge import CvBridge 
from sensor_msgs.msg import Image 
from geometry_msgs.msg import Twist

import numpy as np
import imutils
from imutils import contours, perspective
from random import randint


class video_recording():
  def __init__(self):
    self.uav_camera_subscriber = rospy.Subscriber("/drone/front_cam/front_cam/image",Image,self.uav_video_feed,10)
    self.pub_drone_vel = rospy.Publisher('/drone/cmd_vel', Twist, queue_size=10)
    self.pub_turtle_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    self.uav_out = cv2.VideoWriter('/home/davitmirzoyan/uav_video.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1280,1080))
    self.bridge = CvBridge()

  def uav_video_feed(self, data,a):
      
      frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
      #self.uav_out.write(frame) # saving the video

      image_with_object, turtle_middle_y, turtle_middle_x, box = detect_object(frame)
      image = image_with_object

      if isinstance(image_with_object, bool):
        drone_vel = find_turtle_in_world(False)
        self.pub_drone_vel.publish(drone_vel)
        image = frame
      else:
        drone_vel = find_turtle_in_world(True)
        self.pub_drone_vel.publish(drone_vel)

        turtle_image = image[int(box[0][0]):int(box[0][1]), int(box[0][1]):int(box[2][1])]
        cv2.imshow("turtle",turtle_image)
        cv2.waitKey(1)

        height, width, rgb = image.shape
        frame_middle_x, frame_middle_y = int(height/2), int(width/2)

        image = mark_middle_points(frame_middle_x, frame_middle_y, turtle_middle_x, turtle_middle_y, image)

        x_dir, y_dir = calculate_direction(frame_middle_x, turtle_middle_x, frame_middle_y, turtle_middle_y)
        
        drone_vel = update_drone_velocity(x_dir, y_dir)
        self.pub_drone_vel.publish(drone_vel)

        turtle_vel = turtle_velocity()
        self.pub_turtle_vel.publish(turtle_vel)

      cv2.imshow("Camera's vision", image)
      cv2.waitKey(1) 
        
      


def detect_object(image):
  gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  edges_in_image = cv2.Canny(gray_scale, 50, 100)
  objects_more_visible = cv2.dilate(edges_in_image, None, iterations=1)
  substantive_objects_in_image = cv2.erode(objects_more_visible, None, iterations=1)

  countours_in_image = cv2.findContours(substantive_objects_in_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  countours_in_image = imutils.grab_contours(countours_in_image)

  sorted_contours, _ = contours.sort_contours(countours_in_image)

  count = 0
  print(len(sorted_contours))
  for c in sorted_contours:
    count+=1
    
    # if contour is small, then skip
    if cv2.contourArea(c) < 50:
      continue

    # compute the rotated bounding box of the contour
    box = cv2.minAreaRect(c)


    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    # order the points in the contour such that they appear
    # in top-left, top-right, bottom-right, and bottom-left
    # order, then draw the outline of the rotated bounding
    # box
    box = perspective.order_points(box)
    cv2.drawContours(image, [box.astype("int")], -1, (255, 0, 0), 3)

    original_image = image.copy()
    print(box[0])
    print(box[1])
    print(box[2])
    print(box[3])


    top_line_middle_y = int((box[0][0] + box[1][0]) / 2) 
    top_line_middle_x = int((box[0][1] + box[1][1]) / 2)

    bottom_line_middle_y = int((box[2][0] + box[3][0]) / 2) 
    bottom_line_middle_x = int((box[2][1] + box[3][1]) / 2)

    middle_y_of_box = int((top_line_middle_y + bottom_line_middle_y) / 2)
    middle_x_of_box = int((top_line_middle_x + bottom_line_middle_x) / 2)
    
    if count == len(sorted_contours):
      return image, middle_y_of_box, middle_x_of_box, box
  
  return False, False, False, False # no objects found


def frame_turtle_middle_points(image, box):
  height, width, rgb = image.shape

def find_turtle_in_world(object_found):
  drone_vel = Twist()

  if not object_found:
    drone_vel.angular.z = 0.5
  else:
    drone_vel.angular.z = 0.0

  return drone_vel

def frame_turtle_middle_points(image, box):
  height, width, rgb = image.shape
  frame_middle_x, frame_middle_y = int(height/2), int(width/2)

  top_line_middle_y = int((box[0][0] + box[1][0]) / 2) 
  top_line_middle_x = int((box[0][1] + box[1][1]) / 2)

  bottom_line_middle_y = int((box[2][0] + box[3][0]) / 2) 
  bottom_line_middle_x = int((box[2][1] + box[3][1]) / 2)

  middle_y_of_box = int((top_line_middle_y + bottom_line_middle_y) / 2)
  middle_x_of_box = int((top_line_middle_x + bottom_line_middle_x) / 2)
  
  middle_points_frame_turtle = [frame_middle_x, frame_middle_y, middle_x_of_box, middle_y_of_box]

  return 

def mark_middle_points(frame_middle_x, frame_middle_y, turtle_middle_x, turtle_middle_y, image):
  image[frame_middle_x][frame_middle_y] = 255
  image[frame_middle_x+1][frame_middle_y] = 255
  image[frame_middle_x-1][frame_middle_y] = 255
  image[frame_middle_x][frame_middle_y+1] = 255
  image[frame_middle_x][frame_middle_y-1] = 255

  image[turtle_middle_x][turtle_middle_y] = 255
  image[turtle_middle_x+1][turtle_middle_y] = 255
  image[turtle_middle_x-1][turtle_middle_y] = 255
  image[turtle_middle_x][turtle_middle_y+1] = 255
  image[turtle_middle_x][turtle_middle_y-1] = 255

  return image

def calculate_direction(x1, x2, y1, y2):
  x_direction = x2 - x1

  y_direction = y2 - y1

  return x_direction, y_direction

def update_drone_velocity(x_dir, y_dir):
  drone_vel = Twist()
  print(x_dir, y_dir)

  if abs(x_dir) < 1:
    drone_vel.linear.x = 0
  else:
    if x_dir > 0:
      drone_vel.linear.x = -0.2
    elif x_dir < 0:
      drone_vel.linear.x = 0.2

  if abs(y_dir) < 1:
    drone_vel.linear.y = 0
  else:
    if y_dir > 0:
      drone_vel.linear.y = -0.2
    elif y_dir < 0:
      drone_vel.linear.y = 0.2

  return drone_vel

def turtle_velocity():
  turtle_vel = Twist()
  turtle_vel.linear.x = 0.2

  random_value = randint(0, 20)

  if random_value > 15:
    turtle_vel.angular.z = 0.3
  else:
    turtle_vel.angular.z = 0.0

  return turtle_vel 
  
def main(args=None):
  rospy.init_node('Simulation_Guard')                                           
  video_class_obj = video_recording()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Exiting")
  
if __name__ == '__main__':
  main()