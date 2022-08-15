#!/usr/bin/env python3

import cv2
import rospy
import time
import numpy as np
import os
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState, Joy, BatteryState
from mavros_msgs.msg import OverrideRCIn, RCIn, RCOut
from mavros_msgs.srv import CommandBool


class BlueROV2():
    def __init__(self) -> None:
        pass
    
    def initialize_subscribers(self):
        pass
    
    def initialize_plublishers(self):
        pass
    
    def initialize_timers(self):
        pass
    
    def thrust_to_pwm(self):
        pass
    
    def auto_control(self):
        pass
    
    def manual_control(self):
        pass
    
    def arm(self):
        pass
    
    def disarm(self):
        pass
    
    def controller(self):
        pass
    
    def joy_button_press(self):
        pass
    
    def show_HUD(self):
        pass
    
    def timer_cb(self):
        pass