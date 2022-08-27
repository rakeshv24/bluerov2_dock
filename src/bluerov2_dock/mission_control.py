#!/usr/bin/env python3

import cv2
import rospy
import rospkg
import tf
import time
import numpy as np
import imutils
import sys
import pandas as pd

sys.path.insert(0, '/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/src/bluerov2_dock')

try:
    import video
except:
    import bluerov2_dock.video as video

from auto_dock import MPControl

from std_msgs.msg import Header, Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Joy, BatteryState, FluidPressure
from nav_msgs.msg import Odometry
from mavros_msgs.msg import OverrideRCIn, ManualControl, State
from bluerov2_dock.msg import marker_detect

from mavros_msgs.srv import CommandBool


class BlueROV2():
    def __init__(self) -> None:
        # Set flag for manual or auto mode
        self.mode_flag = 'manual'

        # Set boolean to note when to send a blank button
        # (needed to get lights to turn on/off)
        self.reset_button_press = False

        # Set up pulse width modulation (pwm) values
        self.neutral_pwm = 1500
        self.max_pwm_auto = 1600
        self.min_pwm_auto = 1400
        self.max_pwm_manual = 1700
        self.min_pwm_manual = 1300
        #self.max_possible_pwm = 1900
        #self.min_possible_pwm = 1100
        
        self.override = None
        
        self.rov_odom = None
        self.dock_odom = None
        
        self.first_odom_flag = True
        self.previous_rov_pose = None
        self.rov_pose_sub_time = None
        
        self.first_fid_flag = True
        self.previous_fid_pose = None
        self.fid_pose_sub_time = None
        self.rel_rov_pose = None
        
        self.image_idx = 0
        
        self.load_pwm_lookup()
        
        self.mpc = MPControl()

        # Set up dictionary to store subscriber data
        self.sub_data_dict = {}
        
        # self.setup_video()
        self.initialize_subscribers()
        self.initialize_publishers()
        self.initialize_services()
        # self.initialize_timers()
        
    def load_pwm_lookup(self):
        csv = pd.read_csv("/home/darth/workspace/bluerov2_ws/src/bluerov2_dock/data/T200_data_16V.csv")

        thrust_vals = csv['Force'].tolist()
        neg_thrust = [i for i in thrust_vals if i < 0]
        pos_thrust = [i for i in thrust_vals if i > 0]
        zero_thrust = [i for i in thrust_vals if i == 0]

        pwm_vals = csv['PWM'].tolist()
        neg_t_pwm = [pwm_vals[i] for i in range(len(neg_thrust))]
        zero_t_pwm = [pwm_vals[i] for i in range(len(neg_thrust), len(neg_thrust)+len(zero_thrust))]
        pos_t_pwm = [pwm_vals[i] for i in range(len(neg_thrust)+len(zero_thrust), len(thrust_vals))]

        self.neg_thrusts = np.array(neg_thrust)
        self.pos_thrusts = np.array(pos_thrust)
        self.neg_pwm = np.array(neg_t_pwm)
        self.pos_pwm = np.array(pos_t_pwm)
    
    def initialize_subscribers(self):
        # Set up subscribers
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.store_sub_data, "joy")
        self.battery_sub = rospy.Subscriber('/mavros/battery', BatteryState, self.store_sub_data, "battery")
        self.state_subs = rospy.Subscriber('/mavros/state', State, self.store_sub_data, "state")    
        self.rov_sub = rospy.Subscriber('/qualisys/ROV/odom', Odometry, self.rov_odom_cb)
        self.dock_sub = rospy.Subscriber('/qualisys/Dock/odom', Odometry, self.dock_odom_cb)
        self.fid_sub = rospy.Subscriber('/bluerov2_dock/rel_dock_center', marker_detect, self.fiducial_cb)
        self.pressure_sub = rospy.Subscriber('mavros/imu/static_pressure', FluidPressure, self.pressure_cb)
    
    def initialize_publishers(self):
        # Set up publishers
        # self.control_pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=1)
        self.control_pub = rospy.Publisher('/bluerov2_dock/pwm', OverrideRCIn, queue_size=1)
        self.mpc_output = rospy.Publisher('/bluerov2_dock/mpc', Float32MultiArray, queue_size=1)
        self.lights_pub = rospy.Publisher('/mavros/manual_control/send', ManualControl, queue_size=1)
        self.mpc_rov_odom_pub = rospy.Publisher('/mpc/rov_odom', Float32MultiArray, queue_size=1)
        self.mpc_xr_pub = rospy.Publisher('/mpc/xr', Float32MultiArray, queue_size=1)

    def initialize_services(self):
        # Initialize arm/disarm service
        self.arm_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)        

    def initialize_timers(self):
        rospy.Timer(rospy.Duration(0.05), self.timer_cb)
        
    def wrap_pi2negpi(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi
    
    def fiducial_cb(self, data):
        try:
            self.rel_rov_pose = np.zeros((12,1))
            self.rel_rov_pose[0][0] = -data.locations[0].x
            self.rel_rov_pose[1][0] = -data.locations[0].y
            self.rel_rov_pose[2][0] = -data.locations[0].z
            # self.rel_rov_pose[5][0] = data.locations[0].yaw
            
            if self.first_fid_flag:
                self.fid_pose_sub_time = rospy.Time.now().to_sec()
                self.previous_fid_pose = self.rel_rov_pose
                self.first_fid_flag = False
            else:
                del_time = rospy.Time.now().to_sec() - self.fid_pose_sub_time
                pose_diff = self.rel_rov_pose[0:6, :] - self.previous_fid_pose[0:6, :]
                velocities = pose_diff / del_time
                
                # self.rel_rov_pose[6:12, :] = velocities
                
                self.fid_pose_sub_time = rospy.Time.now().to_sec()
                self.previous_fid_pose = self.rel_rov_pose
            
        except Exception as e:
            rospy.logerr_throttle(10, "[BlueROV2][fiducial_cb] Not receiving fiducial readings" + str(e))
    
    def pressure_cb(self, data):
        try:
            pressure = data.fluid_pressure
            rho = 1000
            g = 9.8
            self.depth = pressure / (rho * g)
        except Exception as e:
            rospy.logerr_throttle(10, "[BlueROV2][pressure_cb] Not receiving pressure readings")
    
    def rov_odom_cb(self, data):
        try:
            x = data.pose.pose.orientation.x
            y = data.pose.pose.orientation.y
            z = data.pose.pose.orientation.z
            w = data.pose.pose.orientation.w
            euler = tf.transformations.euler_from_quaternion((x,y,z,w))
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]
            
            self.rov_pose = np.zeros((6,1))
            self.rov_pose[0][0] = data.pose.pose.position.x
            self.rov_pose[1][0] = data.pose.pose.position.y
            self.rov_pose[2][0] = data.pose.pose.position.z
            self.rov_pose[3][0] = roll
            self.rov_pose[4][0] = pitch
            self.rov_pose[5][0] = yaw
            
            self.rov_pose[3:6, :] = self.wrap_pi2negpi(self.rov_pose[3:6, :])
            
            if self.first_odom_flag:
                self.rov_pose_sub_time = data.header.stamp.to_sec()
                self.previous_rov_pose = self.rov_pose
                self.first_odom_flag = False
                # print("I'm here")
            else:
                del_time = data.header.stamp.to_sec() - self.rov_pose_sub_time
                rov_pose_diff = self.rov_pose - self.previous_rov_pose
                rov_twist = rov_pose_diff / del_time
                
                self.rov_twist = rov_twist
                
                self.rov_pose_sub_time = time.time()
                self.previous_rov_pose = self.rov_pose
                
                self.rov_odom = np.vstack((self.rov_pose, self.rov_twist))
                # print(self.rov_odom)
            
            # self.rov_twist = np.zeros((6,1))
            # self.rov_twist[0][0] = data.twist.twist.linear.x
            # self.rov_twist[1][0] = data.twist.twist.linear.y
            # self.rov_twist[2][0] = data.twist.twist.linear.z
            # self.rov_twist[3][0] = data.twist.twist.angular.x
            # self.rov_twist[4][0] = data.twist.twist.angular.y
            # self.rov_twist[5][0] = data.twist.twist.angular.z
            
            # self.rov_odom = np.vstack((self.rov_pose, self.rov_twist))
            
        except Exception as e:
            rospy.logerr_throttle(10, "[BlueROV2][rov_odom_cb] Not receiving ROV's odometry")
        
    def dock_odom_cb(self, data):
        try:
            x = data.pose.pose.orientation.x
            y = data.pose.pose.orientation.y
            z = data.pose.pose.orientation.z
            w = data.pose.pose.orientation.w
            euler = tf.transformations.euler_from_quaternion((x,y,z,w))
            roll = euler[0]
            pitch = euler[1]
            yaw = euler[2]
            
            self.dock_pose = np.zeros((6,1))
            self.dock_pose[0][0] = data.pose.pose.position.x
            self.dock_pose[1][0] = data.pose.pose.position.y
            self.dock_pose[2][0] = data.pose.pose.position.z
            self.dock_pose[3][0] = roll
            self.dock_pose[4][0] = pitch
            self.dock_pose[5][0] = yaw
            
            self.dock_twist = np.zeros((6,1))
            self.dock_twist[0][0] = data.twist.twist.linear.x
            self.dock_twist[1][0] = data.twist.twist.linear.y
            self.dock_twist[2][0] = data.twist.twist.linear.z
            self.dock_twist[3][0] = data.twist.twist.angular.x
            self.dock_twist[4][0] = data.twist.twist.angular.y
            self.dock_twist[5][0] = data.twist.twist.angular.z
            
            self.dock_odom = np.vstack((self.dock_pose, self.dock_twist))
            
        except Exception as e:
            rospy.logerr_throttle(10, "[BlueROV2][dock_odom_cb] Not receiving dock's odometry")
        
    def store_sub_data(self, data, key):
        """
        Generic callback function that stores subscriber data in the self.sub_data_dict dictionary
        """
        try:
            self.sub_data_dict[key] = data
        except Exception as e:
            rospy.logerr_throttle(10, "[BlueROV2][store_sub_data] Not receiving {} data".format(key))
    
    # def setup_video(self):
    #     # Set up video feed
    #     self.cam = None
    #     self.log_images = False
    #     try:
    #         video_udp_port = rospy.get_param("/mission_control/video_udp_port")
    #         self.log_images = rospy.get_param("/mission_control/log_images")
    #         rospy.loginfo("video_udp_port: {}".format(video_udp_port))
    #         self.cam = video.Video(video_udp_port)
    #     except Exception as e:
    #         rospy.logerr("[BlueROV2][setup_video] Failed to setup video through custom UDP port. Initializing through default port...")
    #         self.cam = video.Video()
    
    def thrust_to_pwm(self, thrust):
        values = []
        thrust = thrust.flatten()
        # thrust[0] = thrust[1]
        # thrust[1] = -thrust[0]
        # thrust[2] = -thrust[2]
        # thrust[5] = -thrust[5]
            
        try:        
            for i in range(6):
                t = thrust[i] / 9.8
                t = np.round(t, 3)
                # print(t)
                
                # p = 0.1427*t**5 - 0.0655*t**4 - 5.6083*t**3 - 0.6309*t**2 + 140.5964*t + 1491.3739
                
                if t > 0.0:
                    p = np.interp(t, self.pos_thrusts, self.pos_pwm)
                elif t < 0.0:
                    p = np.interp(t, self.neg_thrusts, self.neg_pwm)
                else:
                    p = 1500
                
                values.append(round(p))
                
            # pwm = [values[4], values[3], values[2], values[5], values[0], values[1]]
            pwm = [self.neutral_pwm, self.neutral_pwm, values[2], self.neutral_pwm, values[0], values[1]]
            # pwm = [self.neutral_pwm, self.neutral_pwm, values[2], values[5], values[0], values[1]]
            # pwm = [self.neutral_pwm, self.neutral_pwm, self.neutral_pwm, self.neutral_pwm, 1900, self.neutral_pwm]
            
        except Exception as e:
            rospy.logerr_throttle(10, "[BlueROV2][thrust_to_pwm] Error in thrust to pwm conversion. Setting neutral pwm")
            pwm = [self.neutral_pwm for _ in range(6)]
            
        return pwm
    
    def arm(self):
        """ Arm the vehicle and trigger the disarm
        """
        rospy.wait_for_service('/mavros/cmd/arming')
        self.arm_srv(True)
        rospy.loginfo("[BlueROV2][arm] Arming vehicle")

        # Disarm is necessary when shutting down
        rospy.on_shutdown(self.disarm)
    
    def disarm(self):
        rospy.loginfo("[BlueROV2][disarm] Disarming vehicle")
        self.arm_srv(False)
        rospy.wait_for_service('/mavros/cmd/arming')
        self.arm_srv(False)

    def joy_button_press(self, joy_button):
        """Sends button press through mavros/manual_control. Useful for sending
            commands that don't have an explicit rostopic (i.e lights).
        
        Mavros_msgs/ManualControl.msg takes a std_msgs/Header header, 
        float32 x, y, z, and r thruster positions, and uint16 buttons. 
        Thrusters are controlled through rc_override, so are set to zero here.

        Args:
            joy_button (16 integer bitfield): mavlink mapped button, 
                can be set easliy in QGC"""

        # Set up header for mavros_msgs.ManualControl.msg
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "foo"

        # Set thruster values to zero. X, Y, and R normalize from [-1000, 1000].
        # Z has zero set at 500 for legacy reasons.
        x_zero = 0  # forward-back axis
        y_zero = 0  # left-right axis
        z_zero = 500  # up-down axis
        r_zero = 0  # yaw axis

        self.lights_pub.publish(
            header,
            x_zero,
            y_zero,
            z_zero,
            r_zero,
            joy_button
        )

    def controller(self, joy):
        """
        Arbiter to set the ROV in manual or autonomous mode. 
        Sets buttons on the joystick.
        
        Args: 
            joy: joystick subscription
        """            
        axes = joy.axes
        buttons = joy.buttons
        
        # Switch into autonomous mode when button "A" is pressed
        # (Switches back into manual mode when the control sticks are moved)
        if buttons[0]:
            self.mode_flag = 'auto'

        # set arm and disarm (disarm default)
        if buttons[7] == 1:  # "back" joystick button
            self.disarm()
        elif buttons[6] == 1:  # "start" joystick button
            self.arm()

        # turn on lights (can add other button functionality here, 
        # maps to feature set in QGC)
        joy_10 = 0b0000010000000000     # lights_brighter
        joy_9 =  0b0000001000000000     # lights_dimmer
        joy_14 = 0b1000000000000000     # disabled -- MAKE SURE THIS BUTTON IS DISABLED IN QGC

        # If right bumper is pressed
        if buttons[5] == 1:
            # turn lights down one step
            self.joy_button_press(joy_9)
            # set flag so blank button is sent when the lights button is released
            # (this is necessary to get the lights to behave properly)
            self.reset_button_press =  True
        # If left bumper is pressed
        elif buttons[4] == 1:
            # turn light up one step
            self.joy_button_press(joy_10)
            self.reset_button_press =  True
        # If reset flag is true send blank button
        elif self.reset_button_press == True:
            self.joy_button_press(joy_14)
            self.reset_button_press = False

        # set autonomous or manual control (manual control default)
        if self.mode_flag == 'auto':
            self.auto_control(joy)
        else:
            self.manual_control(joy)
    
    def auto_control(self, joy):        
        # Switch out of autonomous mode if thumbstick input is detected
        # Grab the values of the control sticks
        axes = joy.axes
        control_sticks = axes[0:2] + axes[3:5]
        # Check if there is any input on the control sticks
        control_sticks = [abs(val) for val in control_sticks]
        if sum(control_sticks) > 0:
            # Set mode to manual
            self.mode_flag = 'manual'
            # Immediately exit the function
            return
        
        # if self.rov_odom is None:
        #     rospy.logerr_throttle(10, "[BlueROV2][auto_contol] ROV odom not initialized")
        #     return
        
        if self.rel_rov_pose is None:
            rospy.logerr_throttle(10, "[BlueROV2][auto_contol] ROV odom not initialized")
            return
        
        # if self.dock_odom is None:
        #     rospy.logerr_throttle(10, "[BlueROV2][auto_contol] Dock odom not initialized")
        #     return
        
        try:
            # x0 = np.array([[0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
            # xr = np.array([[1., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T
            # x0 = np.array([[-0.21, 0.0, -0.5, 0., 0., 1.57, 0., 0., 0., 0., 0., 0.]]).T
            # xr = np.array([[-0.21, 1.12, -0.5, 0., 0., 1.57, 0., 0., 0., 0., 0., 0.]]).T
            xr = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]).T

            # thrust, converge_flag = self.mpc.run_mpc(x0, xr)
            thrust, converge_flag = self.mpc.run_mpc(self.rel_rov_pose, xr)
            # thrust, converge_flag = self.mpc.run_mpc(self.rov_odom, xr)
            
            # Dear Rakesh: 
            # We added this publishing behavior right here.
            # Sincerely,
            # RDML
            
            
            # thrust, converge_flag = self.mpc.run_mpc(self.rov_odom, self.dock_odom)
        except Exception as e:
            rospy.logerr_throttle(10, "[BlueROV2][auto_control] Error in MPC Computation" + str(e))
            return
        
        if converge_flag:
            rospy.loginfo_throttle(10, "[BlueROV2][auto_control] ROV reached dock successfully! Disarming now...")
            self.disarm()
        else:
            thrust[0:6, :] *= 0.1
            # thrust[5:, 0] *= 0.1
            
            mpc_op = Float32MultiArray()
            dim = MultiArrayDimension()
            dim.label = "MPC Thrusts"
            dim.size = 6
            dim.stride = 1
            mpc_op.layout.dim.append(dim)
            mpc_op.data = [float(thrust[i][0]) for i in range(6)]
            # mpc_op.data.append(thrust[0][0])
            # mpc_op.data.append(thrust[1][0])
            # mpc_op.data.append(thrust[2][0])
            # mpc_op.data.append(thrust[3][0])
            # mpc_op.data.append(thrust[4][0])
            # mpc_op.data.append(thrust[5][0])
            self.mpc_output.publish(mpc_op)
            
            pwm = self.thrust_to_pwm(thrust)
            print(pwm[0:6])

            for _ in range(len(pwm), 18):
                pwm.append(self.neutral_pwm)
            
            for i in range(len(pwm)):
                pwm[i] = max(min(pwm[i], self.max_pwm_auto), self.min_pwm_auto)
                
            print(pwm[0:6])
            print("")

            self.control_pub.publish(pwm)
            
            xr_msg = Float32MultiArray()
            dim1 = MultiArrayDimension()
            dim1.label = "X Ref"
            dim1.size = 12
            dim1.stride = 1
            xr_msg.layout.dim.append(dim1)
            # xr_msg.data = []
            xr_msg.data = [float(xr[i][0]) for i in range(12)]
            self.mpc_xr_pub.publish(xr_msg)

            rov_odom_msg = Float32MultiArray()
            dim2 = MultiArrayDimension()
            dim2.label = "ROV Odom"
            dim2.size = 12
            dim2.stride = 1
            rov_odom_msg.layout.dim.append(dim2)
            # rov_odom_msg.data = []
            # rov_odom_msg.data = self.rov_odom.flatten().tolist()
            rov_odom_msg.data = [float(self.rel_rov_pose[i][0]) for i in range(12)]
            self.mpc_rov_odom_pub.publish(rov_odom_msg)

            # self.override = pwm
        
        # time.sleep(5)
        
        # self.override = [self.neutral_pwm, self.neutral_pwm, self.neutral_pwm, self.neutral_pwm, self.neutral_pwm, self.neutral_pwm]
        # for _ in range(len(self.override), 18):
        #     self.override.append(self.neutral_pwm)
    
        
    def manual_control(self, joy):
        """
        Takes joystick input and sends appropriate pwm value to appropriate channels

        Channels:
            override list elements are pwm values for the channel (element index + 1)
            (i.e. index 0 = channel 1)
            
            1: Pitch (not mapped, BlueROV2 doesn't support pitch anyway)
            2: Roll (not mapped)
            3: Ascend/Descend
            4: Yaw Right/Left
            5: Forward/Backward
            6: Lateral Right/Left
            7: Unused
            8: Camera Tilt Up/Down
        """
        axes = joy.axes
        buttons = joy.buttons
        
        self.mpc.mpc.reset()
        
        # Create a copy of axes as a list instead of a tuple so you can modify the values
        # The RCOverrideOut mesage type also expects a list
        temp_axes = list(axes)
        temp_axes[3] *= -1  # fixes reversed yaw axis
        temp_axes[0] *= -1  # fixes reversed lateral L/R axis

        # Remap joystick commands [-1.0 to 1.0] to RC_override commands [1100 to 1900]
        adjusted_joy = [int(val*300 + self.neutral_pwm) for val in temp_axes]
        # adjusted_joy = [int(val*300 + self.neutral_pwm) for val in axes]
        override = [self.neutral_pwm for _ in range(18)]
        
        # override = [int(val*300 + self.neutral_pwm) for val in axes]
        # for _ in range(len(override), 8):
        #     override.append(0)

        # Remap joystick channels to correct ROV channel 
        joy_mapping = [(0,5), (1,4), (3,3), (4,2), (7,7)]

        for pair in joy_mapping:
            override[pair[1]] = adjusted_joy[pair[0]]

        # Cap the pwm value (limits the ROV velocity)
        for i in range(len(override)):
            override[i] = max(min(override[i], self.max_pwm_manual), self.min_pwm_manual)

        rospy.loginfo("[BlueROV2][manual_control] Joy PWM Values: {}".format(override[0:9]))
        # Send joystick data as rc output into rc override topic
        # print(override[0:9])
        self.control_pub.publish(override)
        # self.override = override
        
    def timer_cb(self, timer_event):
        if self.override is not None:
            self.control_pub.publish(self.override)
    
    def show_HUD(self, frame):
        """
        takes an image as input and overlays information on top of the image

        The following information is added:
            - Battery Voltage (will turn red if below 15.4 volts)
            - Armed Status (Status will show up as red when armed)
            - Mode (Manual/Auto)
            - Small green circle in the center of the screen

        Both modifies and returns input image
        """
        # Try to get voltage and armed status
        try:
            battery_voltage = self.sub_data_dict['battery'].voltage
            armed = self.sub_data_dict['state'].armed
        except Exception as error:
            rospy.logerr_throttle(10, '[BlueROV2][show_HUD] Get data error:' + str(error))

        height, width = frame.shape[:2]
        offset = width // 50

        # Display voltage. If voltage < 15.4 volts, display in red text
        voltage_text_color = (0, 255, 0)
        if battery_voltage < 15.4:
            voltage_text_color = (0, 0, 255)
        battery_voltage = str(round(battery_voltage, 2)) + " V"
        cv2.putText(frame, battery_voltage, (offset, height - offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, voltage_text_color, 2)

        # Display armed/disarmed status
        if not armed:
            armed_text = "Disarmed"
            armed_text_color = (0, 255, 0)
        else:
            armed_text = "Armed"
            armed_text_color = (0, 0, 255)
        cv2.putText(frame, armed_text, (offset, offset + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, armed_text_color, 3)

        # Display mode (manual/auto)
        mode_flag_x = width - offset - len(self.mode_flag * 20)
        cv2.putText(frame, self.mode_flag, (mode_flag_x, offset + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display small green circle at center of screen
        cv2.circle(frame, (width//2, height//2), 8, (0, 255, 0), 2)

        return frame
    
    def run(self):
        """Run user code: activates video stream, grabs joystick data, 
        enables control, allows for optional image logging"""
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            # Try to get video data
            # try:
            #     # Set up video output
            #     frame = self.cam.frame()
            #     frame = cv2.resize(frame, (1280, 720))
            #     # frame = imutils.resize(frame, width=1200)
            # except Exception as error:
            #     rospy.logerr_throttle(10, '[BlueROV2][run] frame error:' + str(error))

            # if frame is not None:
            #     self.show_HUD(frame)
            
            # Try to get joystick axes and button data
            try:
                joy = self.sub_data_dict['joy']
                # Activate joystick control
                self.controller(joy)
            except Exception as error:
                rospy.logerr_throttle(10, '[BlueROV2][run] Controller error:' + str(error))

            # if self.log_images and self.image_idx % 10 == 0:
            #     fname = "img_%04d.png" %self.image_idx
            #     rospack = rospkg.RosPack()
            #     path = rospack.get_path("bluerov2_dock") + "/output/"
            #     self.save(path, fname, frame)

            # # Try to display video feed to screen
            # try:
            #     cv2.imshow('frame', frame)
            #     cv2.waitKey(1)
            # except Exception as error:
            #     rospy.logerr_throttle(10, '[BlueROV2][run] imshow error:' + str(error))

            # self.image_idx += 1
            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node('mission_control', anonymous=True)
    except KeyboardInterrupt:
        rospy.logwarn("Shutting down the node")

    obj = BlueROV2()
    obj.run()
        