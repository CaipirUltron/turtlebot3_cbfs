#!/usr/bin/env python3

import rospy
import numpy as np
from turtlebot_common.Maths import AffineModel
from turtlebot3_cbfs.msg import Model
from geometry_msgs.msg import Pose2D


def set_state(data):
    state = np.array([data.x, data.y, data.theta])
    model.computeModel(state)


if __name__ == '__main__':
    try:
        rospy.init_node('model_broadcaster', anonymous=True)

        pub_freq = rospy.get_param('~rate')
        rate = rospy.Rate(pub_freq)

        model = AffineModel("unicycle")
        model.computeModel(np.zeros(model.model_dim))

        pose_sub = rospy.Subscriber("turtlebot_pose", Pose2D, set_state)
        model_pub = rospy.Publisher('model', Model, queue_size=10)

        while not rospy.is_shutdown():
            model_pub.publish(model.sendModel())
            rate.sleep()

    except rospy.ROSInterruptException:
        pass
