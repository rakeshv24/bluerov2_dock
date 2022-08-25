#!/usr/bin/env python3

import rospy
from mavros_msgs.msg import OverrideRCIn


class PWMPublish():
    def __init__(self) -> None:
        self.control_pub = rospy.Publisher('/mavros/rc/override', OverrideRCIn, queue_size=1)
        self.pwm_sub = rospy.Subscriber('/bluerov2_dock/pwm', OverrideRCIn, self.pwm_cb)
        self.pwm_data = None
        
    def pwm_cb(self, data):
        self.pwm_data = data
    
    def run(self):
        rate = rospy.Rate(50)
        
        while not rospy.is_shutdown():
            if self.pwm_data is not None:
                self.control_pub.publish(self.pwm_data)
            
            rate.sleep()


if __name__ =="__main__":
    try:
        rospy.init_node('pwm_publisher', anonymous=True)
    except KeyboardInterrupt:
        rospy.logwarn("Shutting down the node")

    obj = PWMPublish()
    obj.run()
