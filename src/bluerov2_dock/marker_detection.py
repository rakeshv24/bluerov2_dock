#!/usr/bin/env python3

import yaml
import rospy
import os
import cv2
import math
import numpy as np
import video as video
import threading
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from tf2_ros import TransformException, TransformStamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf.transformations import quaternion_from_euler
import tf2_geometry_msgs
from geometry_msgs.msg import PoseStamped
from bluerov2_dock.msg import marker_pose, marker_detect
from bluerov2_dock.srv import detection
from scipy.spatial.transform import Rotation as R
# import sys
# sys.path.insert(0, '/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/src/bluerov2_dock')


ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}


class Aruco():
    def __init__(self):
        # self.desired_markers = [2, 7, 8, 9, 10, 11]
        self.desired_markers = [1, 4]

        self.marker_size = {1: 0.20,
                            2: 0.10,
                            4: 0.15,
                            7: 0.05,
                            8: 0.10,
                            9: 0.10,
                            10: 0.10,
                            11: 0.10}

        # self.marker_offset = {0: [0.21, 0.006],
        #                       1: [-0.178, 0.012],
        #                       2: [0.09, -0.12],
        #                       3: [-0.123, 0.17],
        #                       4: [0.19, 0.146],
        #                       5: [-0.15, -0.02]}

        self.service_flag = False
        self.bridge = CvBridge()

        # Default marker dictionary
        self.selected_dict = ARUCO_DICT["DICT_6X6_50"]

        # Offset from the ROV's center of gravity (COG) to the camera center
        self.camera_offset = [0.16, -0.06]

        # Provide access to TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.tf_broadcaster = TransformBroadcaster()

        self.initialize_subscribers_publishers()

        self.load_camera_config()

        rospy.Timer(rospy.Duration(0.1), self.marker_detection)

    def load_camera_config(self):
        cwd = os.path.dirname(__file__)
        filename = cwd + '/../../config/in_air/ost.yaml'
        f = open(filename, "r")
        camera_params = yaml.load(f.read(), Loader=yaml.SafeLoader)
        self.cam_mat = np.array(camera_params['camera_matrix']['data'], np.float32).reshape(3, 3)
        self.proj_mat = np.array(camera_params['projection_matrix']['data'], np.float32).reshape(3, 4)
        self.dist_mat = np.array(camera_params['distortion_coefficients']['data'], np.float32).reshape(1, 5)

    def initialize_subscribers_publishers(self):
        self.image_sub = rospy.Subscriber("/BlueROV2/video",
                                          Image, self.callback_image, queue_size=1)
        self.pub = rospy.Publisher('/bluerov2_dock/rel_dock_center', marker_detect, queue_size=1)
        self.image_pub = rospy.Publisher('/bluerov2_dock/marker_detection', Image, queue_size=1)
        self.vision_pose_pub = rospy.Publisher("/bluerov2_dock/vision_pose/pose", PoseStamped, queue_size=1)

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

    def callback_image(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

    def marker_detection(self, timerEvent):
        try:
            frame = self.image
        except Exception as e:
            print("[Aruco][marker_detection] Not receiving any image")
            return

        try:
            prj_mtx = self.proj_mat
            dist_mtx = self.dist_mat
            camera_mtx = self.cam_mat
        except Exception as e:
            print("[Aruco][marker_detection] Not receiving camera info")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(self.selected_dict)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        try:
            # Marker detection
            corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        except BaseException:
            print("[Aruco][marker_detection] Error detecting markers")
            return

        corner = np.array(corners)

        # If at least one marker is detected, then continue
        if ids is not None:

            detected_marker_ids = []
            des_marker_corners = []

            # Loop through the detected markers
            for i, j in enumerate(ids):
                # If the detected marker is not one of the desired markers, then ignore
                if j[0] in self.desired_markers:
                    detected_marker_ids.append(j)
                    des_marker_corners.append(corners[i])

            detected_marker_ids = np.array(detected_marker_ids)
            des_marker_corners = np.array(des_marker_corners)

            cv2.aruco.drawDetectedMarkers(frame, des_marker_corners, detected_marker_ids)

            # Marker Pose Estimation
            for i in range(des_marker_corners.shape[0]):
                marker_id = detected_marker_ids[i][0]
                marker_id_str = "marker_{0}".format(marker_id)
                marker_size = self.marker_size[marker_id]
                # print(i)
                # rospy.loginfo("[Aruco][marker_detection] Markers detected")
                
                marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                            [marker_size / 2, marker_size / 2, 0],
                            [marker_size / 2, -marker_size / 2, 0],
                            [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
                
                _, rvec, tvec = cv2.solvePnP(
                    marker_points, des_marker_corners[i], camera_mtx, dist_mtx)

                rvec = np.array(rvec).T
                tvec = np.array(tvec).T

                # Calculates rodrigues matrix
                rmat, _ = cv2.Rodrigues(rvec)
                # Convert rmat to euler
                # rvec = self.rotation_matrix_to_euler(rmat)
                rvec = R.from_matrix(rmat).as_euler("xyz")

                cv2.drawFrameAxes(frame, camera_mtx, dist_mtx, rvec, tvec, 0.1)
                
                quat = R.from_matrix(rmat).as_quat()
                
                tf_cam_to_marker = TransformStamped()
                tf_cam_to_marker.header.frame_id = "camera_link"
                tf_cam_to_marker.child_frame_id = marker_id_str
                tf_cam_to_marker.header.stamp = rospy.Time.now()
                tf_cam_to_marker.transform.translation.x = tvec[0][0]
                tf_cam_to_marker.transform.translation.y = tvec[0][1]
                tf_cam_to_marker.transform.translation.z = tvec[0][2]
                tf_cam_to_marker.transform.rotation.x = quat[0]
                tf_cam_to_marker.transform.rotation.y = quat[1]
                tf_cam_to_marker.transform.rotation.z = quat[2]
                tf_cam_to_marker.transform.rotation.w = quat[3]
                
                self.tf_broadcaster.sendTransform(tf_cam_to_marker)
                
                # transform lookup: base_link -> map
                try:
                    tf_base_to_map = self.tf_buffer.lookup_transform("map", "base_link", rospy.Time())
                except TransformException as e:
                    rospy.logwarn("[Aruco][marker_detection] Transform unavailable: {0}".format(e))
                    return

                rov_pose = PoseStamped()
                rov_pose.header.frame_id = "map"
                rov_pose.header.stamp = rospy.Time.now()
                rov_pose.pose.position.x = tf_base_to_map.transform.translation.x
                rov_pose.pose.position.y = tf_base_to_map.transform.translation.y
                rov_pose.pose.position.z = tf_base_to_map.transform.translation.z
                rov_pose.pose.orientation.x = tf_base_to_map.transform.rotation.x
                rov_pose.pose.orientation.y = tf_base_to_map.transform.rotation.y
                rov_pose.pose.orientation.z = tf_base_to_map.transform.rotation.z
                rov_pose.pose.orientation.w = tf_base_to_map.transform.rotation.w
                
                # rospy.loginfo("[Aruco][marker_detection] ROV Pose: {0}".format(rov_pose))

                self.vision_pose_pub.publish(rov_pose)
        
        cv2.circle(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)), radius=5, color=(255, 0, 0))
        cv2.imshow("marker", frame)
        cv2.waitKey(1)


    def detect_callback(self, req):
        if req.det_flag:
            self.service_flag = True
            rospy.loginfo("Detection is up and running!")
            return True, "Detection Started"
        else:
            # self.result.release()
            self.service_flag = False
            rospy.loginfo("All operations are suspended!")
            return True, "Detection Stopped"


if __name__ == '__main__':
    rospy.init_node('marker_detection', anonymous=True)
    obj = Aruco()
    d = rospy.Service('bluerov2_dock/control_detection', detection, obj.detect_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down the node")
