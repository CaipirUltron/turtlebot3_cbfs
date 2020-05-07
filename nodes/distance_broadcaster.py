#!/usr/bin/env python3

import rospy
import numpy as np
from turtlebot_common.Maths import Field
from turtlebot3_cbfs.msg import ScalarField
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose2D, PointStamped


if __name__ == '__main__':
    try:
        rospy.init_node('distance_broadcaster', anonymous=True)

        # Control frequency
        while not rospy.is_shutdown():


    except rospy.ROSInterruptException:
        pass