import rospy
import std_msgs.msg
import os

def callback(msg):
    print(msg)

os.environ['ROS_MASTER_URI'] = 'http://10.43.116.24:11311/'
rospy.init_node("hoge")
rospy.loginfo('start')
sub = rospy.Subscriber("sub", std_msgs.msg.String, callback)
pub = rospy.Publisher('bob', std_msgs.msg.Int16, queue_size=10)
rate = rospy.Rate(1)
while not rospy.is_shutdown():
    pub.publish(3)
    rate.sleep()
    print("Published")