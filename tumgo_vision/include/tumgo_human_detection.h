/**
 * FILENAME:    tumgo_human_detection.h
 *
 * DESCRIPTION:
 *
 * This file includes the definition of the TUMgoHumanDetection class which takes in a ROS image (sensor_msgs/Image),
 * converts it to OpenCV format (cv::Mat) and then detects human body shapes in the image.
 *
 *
 * AUTHOR:  Gasper Simonic
 *
 * START DATE: 10.2.2017
 *
 */

#ifndef TUMGO_HUMAN_DETECTION_H
#define TUMGO_HUMAN_DETECTION_H

//C++ related includes.
#include <cstdio>
#include <cmath>
#include <sys/stat.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

// ROS related includes.
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/String.h>

// OpenCV related includes.
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


#include <tumgo_vision/srvImage.h>
#include <tumgo_vision/srvGetImage.h>
#include <tumgo_vision/srvGetCloud.h>
#include <tumgo_vision/srvDetectHuman.h>


// Third party includes for tracking.
// #include "../cf_libs/kcf/kcf_tracker.hpp"
//#include <perception_msgs/Rect.h>

// Self defined includes.
// #include <perception_msgs/Rect.h>

// Debug defines.
// Include this if you want to have visual output.
#define DEBUG


using namespace cv;


/**
 * @brief      Class for human detection.
 */
class TUMgoHumanDetection
{
public:

    /**
     * @brief      Constructor.
     */
    TUMgoHumanDetection(ros::NodeHandle nh);

    /**
     * @brief      Destroys the object.
     */
    ~TUMgoHumanDetection();

    /**
     * @brief      Function for detecting and displaying the human bodies.
     *
     * @param[in]  frame  The frame in which to search for the body.
     */
    void detectAndDisplay(std::string cascade_file);

    /**
     * @brief      Callback for the sensor_msgs::Image.
     *
     * @param[in]  msg   The image in a form of a sensor_msgs::Image.
     */
    void imageCallback(const sensor_msgs::ImageConstPtr &msg);

    /**
     * @brief Updating current image stored in curr_image .
     * @param req.frame     image frame that should be stored
     * @return True if update successful, false otherwise.
     */
    bool updateImageSRV(tumgo_vision::srvImage::Request  &req, tumgo_vision::srvImage::Response &res);

    bool detectHumanSRV(tumgo_vision::srvDetectHuman::Request  &req, tumgo_vision::srvDetectHuman::Response &res);

    /* *******************
     * Callback functions
     * *******************/
    /**
     * @brief Updating current image stored in curr_image .
     * @param msg     image frame that should be stored
     */
    void updateImageCB(const sensor_msgs::ImageConstPtr &msg);


private:
    // Global variables for signaling the image processing.
    static bool m_newImage_static;
    static bool m_newBB_static;

    /// Detect human service server
    ros::ServiceServer detect_human_service;

    /// Parameter for showing raw image and pointcloud
    bool show_raw_;
    /// Parameter for showing processed images
    bool show_images_;
    /// Parameter for synchronizing point cloud with the image
    bool synchronize_;   
    /// Parameter for stating whether we use kinect or robot
    bool robot_;   

    /// Update Image topic subscriber
    ros::Subscriber update_image_sub_; 

    /// Flag for processing the image
    bool process_image_ = false;

    // The node handle
    ros::NodeHandle m_node;

    std::string m_windowName;
    std::string m_directory;
    std::string cascades;

    // Buffer for publishers, subscibers.
    int m_queuesize;

    // Helper member variables for image transformation.
    image_transport::ImageTransport m_it;
    image_transport::Subscriber m_imageSub;

    // Pointer to the cv image.
    cv_bridge::CvImagePtr m_cvPtr;    
    /// Current frame from the camera
    cv::Mat curr_image_;

    /**
     * @brief      Display the image on the screen.
     */
    void displayImage(Mat img);
};





#endif // TUMGO_HUMAN_DETECTION_H
