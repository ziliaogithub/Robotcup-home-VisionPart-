/**
 * main.cpp
 *
 * @author Gasper Simonic <gasper.simonic@tum.de>
 *
 * Description:
 * This is the file that includes the main function of the tumgo_human_detection node.
 *
 */

#include <tumgo_human_detection.h>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "detection_node");

    ros::NodeHandle nh;

    TUMgoHumanDetection node(nh);

    ROS_INFO("Started tumgo_human_detection node.");

    while (ros::ok())
    {
        ros::spinOnce();
    }

    return 0;
}
