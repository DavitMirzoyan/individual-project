#!/usr/bin/python3

import rospy
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist


import imutils
from imutils import contours, perspective

from sklearn.cluster import KMeans
from collections import Counter

import numpy as np
import cv2 

import webcolors
from webcolors import hex_to_rgb

import sys
import time
from matplotlib import pyplot as plt

from random import randint

class identify_track_robot():
  def __init__(self, color):
    rospy.Subscriber("/drone/front_cam/front_cam/image",Image,self.drone_image)
    self.pub_drone_vel = rospy.Publisher('/drone/cmd_vel', Twist, queue_size=10)
    self.pub_turtle_1_vel = rospy.Publisher('/robot_1/cmd_vel', Twist, queue_size=10)
    self.pub_turtle_2_vel = rospy.Publisher('/robot_2/cmd_vel',Twist, queue_size=10)
    self.bridge = CvBridge()

    self.robot_data = ""
    self.drone_vel = Twist()
    self.robot_1_vel = Twist()
    self.robot_2_vel = Twist()

    self.given_color = color
    self.initial_start = True

    self.follow_closest_robot = False
    self.far = False

    self.kmeans_iteration = []

  def drone_image(self, data):
      frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
      image_with_object, all_box, each_box_features, robot_names = self.detect_object(frame)
      image = image_with_object
      
      if isinstance(all_box, bool):
        drone_vel = self.find_turtle_in_world(False)
        self.pub_drone_vel.publish(drone_vel)
        image = frame
      else:
        box_1 = all_box[0]
        box_features_1 = each_box_features[0]

        box_2 = None
        box_features_2 = ''

        if len(all_box) == 2:
          box_2 = all_box[1]
          box_features_2 = each_box_features[1]
        elif len(all_box) == 1:
          box_features_2 = each_box_features[0]

        self.check_which_robot_to_follow(image, box_1, box_2, box_features_1, box_features_2)  
      
      cv2.imshow("Camera's vision", image)      
      cv2.waitKey(1) 

  def check_which_robot_to_follow(self, image, box_1, box_2, box_features_1, box_features_2):

    if self.initial_start:
      image = self.logic_to_follow_drone(image, box_features_1)
      self.initial_start = False
    
    if self.given_color is None:
      image = self.logic_to_follow_drone(image, box_features_1)
    else:
      current_robot_name = "robot_1"
      if box_2 is not None and self.follow_closest_robot:
        box_1 = box_2
        current_robot_name = "robot_2"

      self.robot_data = self.send_values_to_required_object_script(image, box_1, current_robot_name)

      if self.robot_data == 'robot_1':
        image = self.logic_to_follow_drone(image, box_features_1)
        self.follow_closest_robot = False
      elif self.robot_data == 'robot_2' and box_features_1 != box_features_2:
        image = self.logic_to_follow_drone(image, box_features_2)
        self.follow_closest_robot = True
      else:
        drone_vel = self.find_turtle_in_world(False)
        self.pub_drone_vel.publish(drone_vel)

  def logic_to_follow_drone(self, image, box_features):
    
    frame_middle_y, frame_middle_x, turtle_middle_y, turtle_middle_x = self.frame_turtle_middle_points(image, 
                                                                                                       box_features[0])
    image = self.mark_middle_points(frame_middle_y, frame_middle_x, turtle_middle_y, turtle_middle_x, image)
    
    x_dir, y_dir = self.calculate_direction(frame_middle_x, turtle_middle_x, frame_middle_y, turtle_middle_y)
    
    drone_vel = self.update_drone_velocity(x_dir, y_dir, box_features[0])
    self.pub_drone_vel.publish(drone_vel)

    turtle_1_vel = self.turtle_velocity()
    self.pub_turtle_1_vel.publish(turtle_1_vel)

    turtle_2_vel = self.turtle_velocity()
    self.pub_turtle_2_vel.publish(turtle_1_vel)
    
    return image
        
  def detect_object(self, image):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges_in_image = cv2.Canny(gray_scale, 50, 100)
    objects_more_visible = cv2.dilate(edges_in_image, None, iterations=1)
    substantive_objects_in_image = cv2.erode(objects_more_visible, None, iterations=1)

    countours_in_image = cv2.findContours(substantive_objects_in_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    countours_in_image = imutils.grab_contours(countours_in_image)
    sorted_contours, _ = contours.sort_contours(countours_in_image)

    total_count = 0
    count_robot = 0

    all_boxes_found = []
    each_rectangle_features = []

    robot_names = []
    areas = []

    for each_contours in sorted_contours:
      total_count+=1

      if cv2.contourArea(each_contours) < 20:
        continue

      # center of rectangle, width and height of rectangle, rotation angle
      rectangle_features = cv2.minAreaRect(each_contours)

      areas.append(cv2.contourArea(each_contours))

      count_robot += 1
      rectangle_with_corner_points = cv2.boxPoints(rectangle_features)

      rectangle_ordered_corner_points = perspective.order_points(rectangle_with_corner_points).astype("int")
      
      all_boxes_found.append(rectangle_ordered_corner_points)
      each_rectangle_features.append(rectangle_features)

    if len(all_boxes_found) != 0:
      boxes, features = self.remove_detected_objects_inside_robot(all_boxes_found, 
                                                                  each_rectangle_features, 
                                                                  areas)

      robot_count = 0
      final_boxes = []
      finale_features = []

      for i in range(len(boxes)):
        if boxes[i] is not None:
          final_boxes.append(boxes[i])
          finale_features.append(features[i])
          robot_count += 1
          
          robot_name = "Robot {}".format(robot_count)
          robot_names.append(robot_name)

          cv2.putText(image, robot_name, tuple(boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
          cv2.drawContours(image, [boxes[i]], -1, (255, 255, 255), 1)
      
      return image, final_boxes, finale_features, robot_names

    return False, False, False, False # no objects found
  
  def remove_detected_objects_inside_robot(self, all_boxes_found, each_rectangle_feature, each_rectangel_area):

    copy_all_boxes = all_boxes_found.copy()
    copy_each_feature = each_rectangle_feature.copy()

    for i in range(len(all_boxes_found)):
      for j in range(i+1, len(all_boxes_found)):
        center = 0

        if each_rectangel_area[i] < each_rectangel_area[j]:
          contour_center = list(list(each_rectangle_feature[i])[0])
          required_box = all_boxes_found[j]
         
          center = 1

        else:
          contour_center = list(list(each_rectangle_feature[j])[0])
          required_box = all_boxes_found[i]

          center = 2

        if (contour_center[0] > required_box[0][0] and 
          contour_center[0] < required_box[2][0] and
          contour_center[1] > required_box[0][1] and
          contour_center[1] < required_box[2][1]):

          if center == 1:
            copy_all_boxes.pop(i)
            copy_all_boxes.insert(i,None)

            copy_each_feature.pop(i)
            copy_each_feature.insert(i,None)
          elif center == 2:
            copy_all_boxes.pop(j)
            copy_all_boxes.insert(j,None)
            copy_each_feature.pop(j)
            copy_each_feature.insert(j,None)

    return copy_all_boxes, copy_each_feature

  def send_values_to_required_object_script(self, image, box, follow_robot):
    turtle_img = self.crop_robot_image(image, box)
    v = self.checking_image_main_rgb(turtle_img, follow_robot)
    return v

  def crop_robot_image(self, image, box):
    y_1 = int(box[0][1])
    y_2 = int(box[2][1])
    x_1 = int(box[0][0])
    x_2 = int(box[1][0])
    cropped_image = image[y_1:y_2, x_1:x_2]

    return cropped_image 

  def find_turtle_in_world(self, object_found):

    if not object_found:
      self.drone_vel.angular.z = 0.3
      self.drone_vel.linear.x = 0
      self.drone_vel.linear.y = 0
    else:
      self.drone_vel.angular.z = 0.0

    return self.drone_vel

  def frame_turtle_middle_points(self, image, box_center):
    height, width, rgb = image.shape
    frame_middle_y, frame_middle_x = int(height/2), int(width/2)

    middle_y_of_box = int(round(box_center[1]))
    middle_x_of_box = int(round(box_center[0]))

    return frame_middle_y, frame_middle_x, middle_y_of_box, middle_x_of_box

  def mark_middle_points(self, frame_middle_y, frame_middle_x, turtle_middle_y, turtle_middle_x, image):
    image[frame_middle_y][frame_middle_x] = 255
    image[frame_middle_y+1][frame_middle_x] = 255
    image[frame_middle_y-1][frame_middle_x] = 255
    image[frame_middle_y][frame_middle_x+1] = 255
    image[frame_middle_y][frame_middle_x-1] = 255

    image[turtle_middle_y][turtle_middle_x] = 255
    image[turtle_middle_y+1][turtle_middle_x] = 255
    image[turtle_middle_y-1][turtle_middle_x] = 255
    image[turtle_middle_y][turtle_middle_x+1] = 255
    image[turtle_middle_y][turtle_middle_x-1] = 255

    return image

  def calculate_direction(self, x1, x2, y1, y2):
    x_direction = x2 - x1

    y_direction = y2 - y1

    return x_direction, y_direction

  def update_drone_velocity(self, x_dir, y_dir, box_features):
    
    if abs(y_dir) < 30:
      self.drone_vel.linear.x = 0
    else:
      if y_dir > 0:
        self.drone_vel.linear.x = -0.15
      elif y_dir < 0:
        self.drone_vel.linear.x = 0.15

    if abs(x_dir) < 30:
      self.drone_vel.linear.y = 0
    else:
      if x_dir > 0:
        self.drone_vel.linear.y = -0.15
      elif x_dir < 0:
        self.drone_vel.linear.y = 0.15

    if self.far:
      self.drone_vel.linear.y *= 5
      self.drone_vel.linear.x *= 5

    box_features = list(box_features)

    if box_features[0] > 450 and box_features[0] < 830:
      self.drone_vel.angular.z = 0.0
    elif box_features[0] < 450:
      self.drone_vel.angular.z = 0.3
    elif box_features[0] > 830:
      self.drone_vel.angular.z = -0.3
    
    return self.drone_vel

  def turtle_velocity(self):
    turtle_vel = self.robot_1_vel
    turtle_vel.linear.x = 0.15

    random_value = randint(0, 22)

    if random_value > 15:
      random_right_or_left = randint(0,2)

      if random_right_or_left == 0:
        turtle_vel.angular.z = 0.15
      elif random_right_or_left == 1:
          turtle_vel.angular.z = -0.15
    else:
      turtle_vel.angular.z = 0.0

    self.robot_1_vel = turtle_vel

    return turtle_vel 

  def checking_image_main_rgb(self, image, follow_robot):
    start = time.time()
    required_color = self.given_color
    try:
              
      robot_image = image.reshape(image.shape[0]*image.shape[1], 3)

      kmeans = KMeans(n_clusters = 3, max_iter=50, n_init = 1)
      find_and_predict_color = kmeans.fit_predict(robot_image)

      quantity_of_each_color = Counter(find_and_predict_color)
      center_of_colors = kmeans.cluster_centers_

      most_found_colors_value = list(quantity_of_each_color.values())
      copy_list = most_found_colors_value.copy()

      max_value = max(most_found_colors_value)
      index_of_max_color = most_found_colors_value.index(max_value)

      copy_list[index_of_max_color] = 0
      index_of_second_max_value = copy_list.index(max(copy_list))

      rgb_colors = [center_of_colors[i] for i in quantity_of_each_color.keys()]
      rgb_names = []
    
      hex = []
      for found_rgb in rgb_colors:
        found_rgb = np.around(found_rgb).astype(int)  

        hex.append(self.rgb_to_hex(tuple(found_rgb)))
        rgb_names.append(self.find_color_name(found_rgb.tolist()))
              
      follow = ''
      another_color = ''
      
      if required_color == 'black':
        follow = required_color
        another_color = 'gray'
      elif required_color == 'gray':
        follow = required_color
        another_color = 'black'
      else:
        follow = 'No such robot'

      self.far = False

      #sometimes when the robot is too far the 1st main or 2nd main 
      #color identified as dimgray instead of gray, (rgb values are close to each other)
      
      if rgb_names[index_of_max_color] == "dimgray":
        rgb_names[index_of_max_color] = "gray"
      
      elif rgb_names[index_of_second_max_value] == "dimgray":
        rgb_names[index_of_second_max_value] = "gray"
      
      # when the robot is too far the main color is identified as background color, 
      # so we need to check the 2nd main color
      if rgb_names[index_of_max_color] == "darkslategray":
        rgb_names[index_of_max_color] = rgb_names[index_of_second_max_value]

      end = time.time()

      '''
      print(len(self.kmeans_iteration))
      if len(self.kmeans_iteration) == 50:
        f = plt.figure()
        f.set_size_inches(10,10)
        ax1 = f.add_subplot(1,2,1)
        ax1.set_title("kmeans execution time")
        plt.plot(self.kmeans_iteration, label="n_init: 1")
        plt.plot(ten, label = "n_init: 10")
        plt.savefig("timing", dpi=300)
        rospy.signal_shutdown("Task Completed")

      self.kmeans_iteration.append(end - start)
      print(rgb_names[index_of_max_color])
      '''
      if follow == 'No such robot':
        return "wrong"  
      if rgb_names[index_of_max_color] == follow and follow_robot == "robot_2":
        return "robot_2"
      elif rgb_names[index_of_max_color] == follow and follow_robot == "robot_1":
        return "robot_1"
      elif rgb_names[index_of_max_color] == another_color and follow_robot == "robot_1":
        return "robot_2"
      elif rgb_names[index_of_max_color] == another_color and follow_robot == "robot_2":
        return "robot_1"

      return "wrong"
      
    except:
      pass

  def find_color_name(self, given_rgb):
    given_red = given_rgb[0]
    given_green = given_rgb[1]
    given_blue = given_rgb[2]

    all_values = []
    for hex_value, color_name in webcolors.CSS3_HEX_TO_NAMES.items():
      red, green, blue = hex_to_rgb(hex_value)

      #finding closest color name in RGB space

      red_distance = pow((red - given_red), 2)
      green_distance = pow((green - given_green), 2)
      blue_distance = pow((blue - given_blue), 2)

      each_pair = [red_distance+green_distance+blue_distance, color_name]
            
      all_values.append(each_pair)

    all_distances = [each_pair[0] for each_pair in all_values]

    index_of_min_distance = all_distances.index(min(all_distances))
    name_of_min_distance = all_values[index_of_min_distance][1]
        
    return name_of_min_distance
  
  def rgb_to_hex(self, color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

  def checking_image_size(self, image):
    height, width, rgb = image.shape

    if height > width:
      return "gray"

    return "black"

def main(args=None):
  argv = sys.argv[1:]

  if len(argv) == 1:
    color = argv[0]
  else:
    color = None
  
  rospy.init_node('Identify_Track_Object')                                           
  identify_track_robot(color)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Exiting")
  
if __name__ == '__main__':
  main()