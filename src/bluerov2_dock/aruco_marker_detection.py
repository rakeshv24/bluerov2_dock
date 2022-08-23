#!/usr/bin/env python3

import rospy
import yaml
import cv2
import math
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import sys

sys.path.insert(0, '/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/src/bluerov2_dock')

from bluerov2_dock.msg import marker_pose, marker_detect
from bluerov2_dock.srv import detection

    
ARUCO_DICT = {
    # "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    # "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    # "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    # "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    # "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    # "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    # "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    # "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    # "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    # "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    # "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    # "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    # "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    # "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    # "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    # "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    # "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    # "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    # "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    # "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }
    

class Aruco():
    def __init__(self):
        self.service_flag = False
        self.bridge = CvBridge()
        self.first_marker = True
        self.max_size = -10000
        self.marker_thresh = 25
        
        self.desired_markers = [0, 1, 2, 3, 4, 5]
        self.marker_size = {0: 0.30,
                            1: 0.20,
                            2: 0.10,
                            3: 0.05,
                            4: 0.15,
                            5: 0.25}
        self.marker_offset = {0: [],
                              1: [],
                              2: [0.0675, -0.08],
                              3: [-0.04875, 0.055],
                              4: [0.075, 0.0575],
                              5: [-0.065, -0.0575]}
        self.camera_offset = [-0.16, 0.06]
        self.dock_center_offset = 0.32
        
        self.prev_marker = 10000
        self.counter = 0
        # self.counter1 = 0
        
        # self.result = cv2.VideoWriter('/home/darth/detection.avi', 
        #             cv2.VideoWriter_fourcc(*'MJPG'),
        #             10, (1280, 960))
        self.load_camera_config()
        self.initialize_subscribers_publishers()
        rospy.Timer(rospy.Duration(0.1), self.marker_detection)
        
    def load_camera_config(self):
        filename = '/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/config/in_water/calibrationdata/ost.yaml'
        f = open(filename, "r")
        camera_params = yaml.load(f.read(), Loader=yaml.SafeLoader)
        self.cam_mat = np.array(camera_params['camera_matrix']['data'], np.float32).reshape(3, 3)
        self.proj_mat = np.array(camera_params['projection_matrix']['data'], np.float32).reshape(3, 4)
        self.dist_mat = np.array(camera_params['distortion_coefficients']['data'], np.float32).reshape(1, 5)
    
    def initialize_subscribers_publishers(self):
        self.image_sub = rospy.Subscriber("/BlueROV2/video",
                                          Image, self.callback_image, queue_size=1)
        self.pub = rospy.Publisher('/bluerov2_dock/marker_locations', marker_detect, queue_size=1)
        self.image_pub = rospy.Publisher('/bluerov2_dock/marker_detection', Image, queue_size=1)

    def callback_image(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

    def rotation_matrix_to_euler(self, R):
        def isRotationMatrix(R):
            Rt = np.transpose(R)
            shouldBeIdentity = np.dot(Rt, R)
            I = np.identity(3, dtype=R.dtype)
            n = np.linalg.norm(I - shouldBeIdentity)
            return n < 1e-6
        assert (isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z], np.float32)

    def marker_detection(self, timerEvent):
        try:
            frame = self.image
        except Exception as e:
            rospy.logerr_throttle(10, "Not receiving any image")
            return

        try:
            prj_mtx = self.proj_mat
            dist_mtx = self.dist_mat
            camera_mtx = self.cam_mat
        except Exception as e:
            rospy.logerr_throttle(10, "Not receiving camera info")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        pub_obj = marker_detect()

        if self.service_flag == True:
            
            # Loop through all the different dictonaries for the aruco markers
            for (aruco_name, aruco_dict) in ARUCO_DICT.items():
                
                aruco_dict = cv2.aruco.Dictionary_get(aruco_dict)
                aruco_params = cv2.aruco.DetectorParameters_create()

                try:
                    # Marker detection
                    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(
                        gray, aruco_dict, parameters=aruco_params)
                except BaseException:
                    rospy.logerr_throttle(10, "Error detecting markers")
                    return

                corner = np.array(corners)

                # If at least one marker is detected, then continue
                if ids is not None:

                    max_marker_index = -1
                    des_detected_markers = []
                    detected_marker_ids = []
                    self.prev_marker_check = False
                    self.prev_marker_index = 10000

                    # Loop through the detected markers
                    for i, j in enumerate(ids):
                        # If the detected marker is not one of the desired markers, then ignore
                        if j[0] not in self.desired_markers:
                            continue
                        
                        # If the current detection is the same as in the previous frame, then assert the flag and store 
                        # the previous marker index
                        if j[0] == self.prev_marker:
                            self.prev_marker_check = True
                            self.prev_marker_index = i

                        # Extract the side length of the marker in pixels
                        side_length = abs(corners[i][0][0][0] - corners[i][0][2][0]) + \
                            abs(corners[i][0][0][1] - corners[i][0][2][1])

                        # Checks if the side length of the current marker is greater than a threshold (which accounts 
                        # for the detection of the smallest marker), then store that information                 
                        check_thresh = side_length > self.marker_thresh
                        if check_thresh:
                            detected_marker_ids.append(i)
                            des_detected_markers.append(self.marker_size[j[0]])
                        
                        # Checks if the size of the current marker is greater than that of the largest marker from the 
                        # previous frames, then store the current marker as the largest marker
                        check_max = self.marker_size[j[0]] >= self.max_size
                        if check_max:
                            self.max_size = self.marker_size[j[0]]
                            max_marker_index = i

                    # If there are more than one of the desired markers, pick the smallest marker; 
                    # Else, pick the largest detectable marker one or report that none of 
                    # the desired markers were detected.
                    if len(des_detected_markers) > 0:
                        min_index = des_detected_markers.index(min(des_detected_markers))
                        target_index = detected_marker_ids[min_index]
                    else:
                        if max_marker_index == -1:
                            target_index = -1
                        else:
                            target_index = max_marker_index

                    if target_index == -1:
                        pub_obj.locations = []
                        pub_obj.detection_flag = False
                        pub_obj.header = Header()
                        pub_obj.header.stamp = rospy.Time.now()
                        self.pub.publish(pub_obj)
                        return None
                    else:
                        # If it is the very first marker detection, then store that as the previous marker info as well
                        if self.first_marker:
                            self.prev_marker = ids[target_index][0]
                            self.counter = 0
                            pub_obj.controller_param = True
                            self.first_marker = False

                        else:
                            # If the previous marker and the current marker are the same, 
                            # then assert the controller flag
                            if self.prev_marker == ids[target_index][0]:
                                pub_obj.controller_param = True
                                target_index = self.prev_marker_index
                                self.counter = 0
                            # Else, perform marker switching
                            else:
                                # Check if the smaller marker remains the same for 5 frames before switching
                                if self.counter > 5.0:
                                    # self.counter1 = 1
                                    self.prev_marker = ids[target_index][0]
                                    self.counter = 0
                                else:
                                    # Check if it remains the same
                                    if self.prev_marker_check:
                                        #pub_obj.marker_switch = True
                                        pub_obj.controller_param = True
                                        target_index = self.prev_marker_index
                                        self.counter += 1
                                    # If not, return none
                                    else:
                                        pub_obj.locations = []
                                        pub_obj.detection_flag = False
                                        pub_obj.header = Header()
                                        pub_obj.header.stamp = rospy.Time.now()
                                        self.pub.publish(pub_obj)
                                        return None
                                            
                    # Marker Pose Estimation
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners[target_index], self.marker_size[ids[target_index][0]], camera_mtx, dist_mtx)

                    # cv2.aruco.drawDetectedMarkers(frame, corners[index])
                    cv2.aruco.drawAxis(frame, camera_mtx, dist_mtx, rvec, tvec, 2)

                    # Calculates rodrigues matrix
                    rmat = cv2.Rodrigues(rvec)[0]
                    # Convert rmat to euler
                    rvec = self.rotation_matrix_to_euler(rmat)
                    
                    # Relative Position
                    # TODO: Fix the orientation of the translation and rotational vecs
                    body_frame = np.zeros(shape=(1, 3))
                    # body_frame[0][0] = tvec[0][0][0] - self.marker_offset[ids[target_index][0]][0] - self.camera_offset[0]
                    # body_frame[0][1] = tvec[0][0][1] - self.marker_offset[ids[target_index][0]][1] - self.camera_offset[1]
                    # body_frame[0][2] = tvec[0][0][2] - self.marker_offset[ids[target_index][0]][2] - self.camera_offset[2] 
                    body_frame[0][0] = tvec[0][0][0]
                    body_frame[0][1] = tvec[0][0][1]
                    body_frame[0][2] = tvec[0][0][2]
                    
                    # if self.counter1 > 0 and self.counter1 < 20:
                    #     pub_obj.marker_switch = True
                    #     self.counter1 += 1

                    pub_obj.locations = []
                    p = marker_pose()
                    p.x = body_frame[0][0]
                    p.y = body_frame[0][1]
                    p.z = body_frame[0][2]
                    p.roll = rvec[0]
                    p.pitch = rvec[1]
                    p.yaw = rvec[2]
                    pub_obj.locations.append(p)
                    pub_obj.detection_flag = True
                    pub_obj.header = Header()
                    pub_obj.header.stamp = rospy.Time.now()
                    self.pub.publish(pub_obj)

                else:
                    pub_obj.locations = []
                    pub_obj.detection_flag = False
                    pub_obj.header = Header()
                    pub_obj.header.stamp = rospy.Time.now()
                    self.pub.publish(pub_obj)

                cv2.imshow("marker", frame)
                # self.result.write(frame)
                # publish_image = Image()
                # publish_image.data = frame
                ros_image = self.cvbridge.cv2_to_imgmsg(frame, 'bgr8')
                self.image_pub.publish(ros_image)
                cv2.waitKey(1)

            else:
                pub_obj.locations = []
                pub_obj.detection_flag = False
                pub_obj.header = Header()
                pub_obj.header.stamp = rospy.Time.now()
                self.pub.publish(pub_obj)

    def detect_callback(self, req):

        if req.det_flag:
            self.service_flag = True
            rospy.loginfo("Detection is up and running!")
            return True, "Detection Started"
        else:
            self.result.release()
            self.service_flag = False
            rospy.loginfo("All operations are suspended!")
            return True, "Detection Stopped"


if __name__ == '__main__':
    rospy.init_node('detection', anonymous=True)
    obj = Aruco()
    d = rospy.Service('bluerov2_dock/control_detection', detection, obj.detect_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down the node")
        