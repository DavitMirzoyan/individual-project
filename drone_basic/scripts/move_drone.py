#!/usr/bin/python3

import rospy
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from geometry_msgs.msg import Twist
import tf

from cv_bridge import CvBridge 
import sensor_msgs.point_cloud2 as pc2
import image_geometry
from image_geometry import PinholeCameraModel

import time
from matplotlib import pyplot as plt
import cv2
import numpy as np
import imutils
from imutils import contours, perspective

import torch
from torchvision import transforms
import torch.nn as nn
from torch.nn import Module
from PIL import Image as img


import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class Encoder_Decoder(Module):
    
    def __init__(self):
        super(Encoder_Decoder, self).__init__()

        self.encoder = nn.Sequential(
            
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
           
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(128, 64, 3, stride=2, 
                                padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, 
                                padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, 3, 3, stride=2, 
                                padding=1, output_padding=1),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )

    def forward(self,high_res_image):
        
        low_res_image = self.encoder(high_res_image)
        reconstructed_image = self.decoder(low_res_image)

        return reconstructed_image

class avoiding_walls():
    def __init__(self):
        rospy.Subscriber("drone/kinect/kinect/depth/points", PointCloud2, self.point_cloud)
        self.point_depth = PointCloud2()

        rospy.Subscriber("drone/kinect/kinect/rgb/camera_info", CameraInfo, self.camera_info)
        self.c_info = CameraInfo()     
        self.model = PinholeCameraModel()   
        
        rospy.Subscriber("drone/kinect/kinect/rgb/image_raw", Image, self.follow_robot)

        self.update_drone_vel = rospy.Publisher('/drone/cmd_vel', Twist, queue_size=10)
        self.c = rospy.Subscriber("drone/cmd_vel", Twist, self.drone_current_vel)
        self.drone_vel = Twist()

        self.bridge = CvBridge()
        self.tf_listener = tf.TransformListener()
        self.drone_pos = tf.TransformListener()

        self.last_pixel_y = -1

        self.last_x_coord = 0
        self.last_y_coord = 0
        
        self.client = actionlib.SimpleActionClient('move_base_drone',MoveBaseAction)

        self.convert_to_tensor = transforms.ToTensor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder_decoder = torch.load('catkin_ws/src/drone_basic/scripts/model.pth', map_location={'cuda:0': 'cpu'})

        self.count = 0
        self.history = []
    
    def drone_current_vel(self, data):
        self.drone_vel = data

    def camera_info(self, data):
        self.c_info = data
        cam_model = image_geometry.PinholeCameraModel()
        cam_model.fromCameraInfo(self.c_info)
        self.model = cam_model

    def point_cloud(self, data):
        self.point_depth = data

    def follow_robot(self, data):
        frame = self.bridge.imgmsg_to_cv2(data,'bgr8')
        #colored_image = self.load_model(frame) # uncomment this line to use model

        colored_image = frame # remove this line if you want to use model
        x_pixel, y_pixel = self.find_robot_points(colored_image)
        
        if isinstance(x_pixel, bool):
            self.drone_pos.waitForTransform('/map', 
                                            '/base_link', 
                                            rospy.Time(), 
                                            rospy.Duration(1))
            (transform, rotation) = self.drone_pos.lookupTransform('/map', 
                                                                   '/base_link', 
                                                                   rospy.Time())
            transform = list(transform)
                  
            x_diff = abs(transform[0] - self.last_x_coord)
            y_diff = abs(transform[1] - self.last_y_coord)

            print(x_diff, " x ", y_diff, " y")

            if x_diff < 0.5 and y_diff < 0.5:
                self.client.cancel_all_goals()
                if self.last_pixel_y > 880:
                    self.drone_vel.linear.x = 0
                    self.drone_vel.linear.y = 0
                    self.drone_vel.angular.z = 2
                else:
                    self.drone_vel.angular.z = 2
            else:
                self.drone_vel.linear.x *= 1.4  
                self.drone_vel.linear.y *= 1.4
                self.drone_vel.angular.z *= 1.1
            self.update_drone_vel.publish(self.drone_vel) 
           
        else:
            if self.drone_vel.linear.x == 0:
                self.drone_vel.linear.x = 0.3
            else:
                self.drone_vel.linear.x *= 1.3     

            self.update_drone_vel.publish(self.drone_vel) 

            depth = pc2.read_points(self.point_depth, skip_nans=False, 
                                     field_names="z", uvs=[(x_pixel, y_pixel)])
            depth = list(next(depth))
            ray = self.model.projectPixelTo3dRay((x_pixel, y_pixel))
            camera_depth = np.concatenate( (depth[0]*np.array(ray), np.ones(1))).reshape((4, 1))
            
            self.last_pixel_y = y_pixel

            self.tf_listener.waitForTransform('/map', 
                                              '/kinect_depth_optical_frame', 
                                              rospy.Time(), rospy.Duration(1))

            (transform, rotation) = self.tf_listener.lookupTransform('/map', 
                                                                     '/kinect_depth_optical_frame', 
                                                                     rospy.Time())
            camera_to_base = tf.transformations.compose_matrix(translate=transform, 
                                                        angles=tf.transformations.euler_from_quaternion(rotation))

            coordinates = np.dot(camera_to_base, camera_depth)
            
            x = coordinates[0]
            y = coordinates[1]

            self.last_x_coord = x
            self.last_y_coord = y

            self.client.wait_for_server()
    
            position = MoveBaseGoal()
            position.target_pose.header.frame_id = "map"
            position.target_pose.header.stamp = rospy.Time.now()
            position.target_pose.pose.position.x = x
            position.target_pose.pose.position.y = y
            position.target_pose.pose.position.z = 0
            position.target_pose.pose.orientation.w = 1

            self.client.send_goal(position)

    def load_model(self, image):
        '''
        # To count how long it takes to receive image from model
        start = time.time() 
        '''
        # pillow image must be in 'PIL.PngImagePligin.PngImageFile' type
        # that is why I saved the image and then opned it with Pillow Image
        
        self.count += 1
        cv2.imwrite("pillow.png", image)
        image = img.open("pillow.png")
        image = self.convert_to_tensor(image)
        
        colored_image = self.encoder_decoder(image[None,...].to(self.device))
        colored_image = (transforms.ToPILImage()(colored_image[0])).convert("RGB") 
        
        colored_image = np.array(colored_image)
        
        '''
        getting graph of model's output time 
        end = time.time()
        self.history.append(end - start)
        
        if len(self.history) == 20:
                f = plt.figure()
                f.set_size_inches(10,10)
                ax1 = f.add_subplot(1,2,1)
                ax1.set_title("geting image from model")
                plt.plot(self.history)
                plt.savefig("model_timing", dpi=300)
                rospy.signal_shutdown("Task Completed")
        '''

        return colored_image

    def find_robot_points(self, colored_image):
        white_robot = self.find_black_color(colored_image)
        x, y = self.crop_robot(white_robot)

        return x, y

    def find_black_color(self, image):
        #boundaries = [([235, 235, 235], [255, 255, 255])] # for model
        boundaries = [([0, 0, 0], [2, 2, 2])] # for alternative solution

        for (lower, upper) in boundaries:
            lower = np.array(lower, dtype = "uint8")
            upper = np.array(upper, dtype = "uint8")

            mask = cv2.inRange(image, lower, upper)
            #image = cv2.bitwise_and(image, image, mask = mask) # for model
            image = cv2.bitwise_not(mask) # for alternative solution

            #cv2.imwrite("input_images/b{}.png".format(self.count), image)
            return image

    def crop_robot(self, image): 
        edges_in_image = cv2.Canny(image, 50, 100)
        objects_more_visible = cv2.dilate(edges_in_image, None, iterations=1)
        substantive_objects_in_image = cv2.erode(objects_more_visible, 
                                                None, iterations=1)
        countours_in_image = cv2.findContours(substantive_objects_in_image, 
                                                         cv2.RETR_EXTERNAL, 
                                                         cv2.CHAIN_APPROX_SIMPLE)
        countours_in_image = imutils.grab_contours(countours_in_image)

        if len(countours_in_image) == 0:
            return False, False

        sorted_contours, _ = contours.sort_contours(countours_in_image)
        count = 0
        
        for i in sorted_contours:
            count+=1
            #print(cv2.contourArea(i))

            if cv2.contourArea(i) < 2000:
                if count != len(sorted_contours):
                    continue
                else:
                    return False, False
            
            rectangle_f = cv2.minAreaRect(i)

            mid_poits = list(rectangle_f)[0]

            x = mid_poits[0]
            y = mid_poits[1]

            return int(x), int(y)
        return False, False

def main(args=None):
    rospy.init_node('find_position')
    avoiding_walls()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Exit")
if __name__ == '__main__':
    main()