#ifndef TUMGO_VISION2D
#define TUMGO_VISION2D

/** @file tumgo_vision2D.h
 *  @brief File storing declaration of the 2D vision functions.
 */

// ROS library
#include <ros/ros.h>
#include <ros/duration.h>

// Helper functions
#include <helpers.h>

// Standard libraries
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <vector>
#include <algorithm>
#include "Eigen/Core"

// Services
#include <tumgo_vision/srvDetectFace.h>
#include <tumgo_vision/srvDetectColorBlob.h>
#include <tumgo_vision/srvTrackObject.h>
#include <tumgo_vision/srvImage.h>
#include <tumgo_vision/srvGetImage.h>

// Messages
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

//thread
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/date_time.hpp>
#include <boost/thread/locks.hpp>
#include <mutex>
#include <thread>
#include <pthread.h>

#define SHOW_RAW true
#define SHOW_IMAGES false

using namespace std;
using namespace cv;

/// Vision class storing all services and messages for 2D vision
class TUMgoVision2D{
public:
    // Constructors
    /// Constructor
    TUMgoVision2D(ros::NodeHandle n);
    /// Destructor
    ~TUMgoVision2D(){};

    /* ************************ *
     * Preprocessing functions  *
     * ************************ */
    /**
    * CLAHE algorithm.
    * @param img            image which histogram should be normalized, Mat type.
    * @param ClipLimit      algorithm parameter on which the "grain" depends, float type.
    * @param TilesGridSize  algorithm parameter on which the size of the tiles depends, float type.
    */
    Mat clahe(Mat img, float ClipLimit, int TilesGridSize);

    /**
    * Function rotating the image.
    * @param img            image that will be rotated, Mat type,
    * @param center         center of the rotation, Point2f type.
    */
    Mat rotate(Mat img,Point2f center, double angle);

    /**
    * Filtering algorithm.
    * @param    img            image which should be blurred, Mat type.
    * @param    type           type of the blur used, 0 - Gaussian, 1 - Median blur, 2 - Bilateral Filter, 3 - Adaptive BF, 4 - Blur, 5 - Dilate, 6 - Erode.
    * @return   Filtered image frame
    */
    Mat filterImage(Mat img, int type);

    /**
    * Extraction of the color.
    * @param    img            image in which the color blobs should be extracted, Mat type.
    * @param    color          color that should be extracted
    * @return   Extracted color blob.
    */
    vector<cv::Point> extractColor(Mat img, string color);

    /**
    * Updating image on request
    * @return   True if image updated, False otherwise.
    */
    bool getImage(void);

    bool startTracking(Rect trackWindow);

    Rect tracking(Rect track_window);

    /* *******************
     * Service functions
     * *******************/   
    /**
     * @brief Extraction of the biggest color blob in the frame.
     * @return Returns valid color blob center with x-,y- position and color or empty object with the flag valid set to false.
     */
    bool detectColorBlobSRV(tumgo_vision::srvDetectColorBlob::Request  &req, tumgo_vision::srvDetectColorBlob::Response &res);

    /**
     * @brief Processing the image and recognize face
     * @param req:void
     * @param res.result(bool):true if we detect successfully, else false
     * @param res.name(string[]):name of face which is detected
     * @param res.face_count(int32):the number of face were detected
     * @param res.x,res.y(int32p[]): the 2D postion of left up point of face in 2d image
     * @param res.width,res.height(int32[]): the size of face
     * @return
     */
    bool detectFaceSRV(tumgo_vision::srvDetectFace::Request  &req, tumgo_vision::srvDetectFace::Response &res);
 
    /**
     * @brief Tracking the object using backprojection
     * @param req.mean      Mean/Cam shift used
     * @param req.x         X-axis position of the tracking
     * @param req.y         X-axis position of the tracking
     * @param req.width     width of the track window
     * @param req.height    height of the track window
     * @param res.valid     whether the window is correct
     * @param res.x         X-axis position of the tracking
     * @param res.y         X-axis position of the tracking
     * @param res.width     width of the track window
     * @param res.height    height of the track window
     * @return
     */   
    bool trackObjectSRV(tumgo_vision::srvTrackObject::Request  &req, tumgo_vision::srvTrackObject::Response &res);
    /**
     * @brief Updating current image stored in curr_image .
     * @param req.frame     image frame that should be stored
     * @return True if update successful, false otherwise.
     */
    bool updateImageSRV(tumgo_vision::srvImage::Request  &req, tumgo_vision::srvImage::Response &res);
    
    /* *******************
     * Callback functions
     * *******************/  
    /**
     * @brief Updating current image stored in curr_image .
     * @param msg     image frame that should be stored
     */
    void updateImageCB(const sensor_msgs::ImageConstPtr &msg);

private:
    // Services
    /// Detect Color Blob service server
    ros::ServiceServer detect_color_blob_service_;
    /// Detect Face service server
    ros::ServiceServer detect_face_service_;
    /// Track Object service server
    ros::ServiceServer track_object_service_;
    /// Update Image service server
    ros::ServiceServer update_image_service_;

    /// Get Image service client
    ros::ServiceClient get_image_client_;

    /// Update Image topic subscriber
    ros::Subscriber update_image_sub_;

    /// The node handle
    ros::NodeHandle nh_;
    /// Node handle in the private namespace
    ros::NodeHandle priv_nh_;

    //for saving the centroid position, width and height of the object
    /// Centroid of the detected object
    Eigen::Vector4f centroid_obj_;
    /// Width of the detected object
    float width_;
    /// Height of the detected object
    float height_;

    /// Current frame from the camera
    cv::Mat curr_image_;
    sensor_msgs::Image curr_msg_;

    /// Lock for camera frame
    pthread_mutex_t image_mutex_ = PTHREAD_MUTEX_INITIALIZER;
    /// Flag for processing the image
    bool process_image_ = false;

    /// Parameter for showing raw image and pointcloud
    bool show_raw_;
    /// Parameter for showing processed images
    bool show_images_;
    /// Parameter for showing extracted colors
    bool show_colors_;
    /// Parameter for synchronizing point cloud with the image
    bool synchronize_; 
    /// Parameter for stating whether we use kinect or robot
    bool robot_;   
    ///for saving the path of cascades files
    std::string cascades_pth_;

    // TRACKER
    const int histSize_ = 30;
    float hranges_[2] = { 0, 180};
    const float* ranges_ = {hranges_};
    MatND hist_target_;
    bool tracking_ = false;
    bool mean_ = false;
    Rect startTrackWindow_;
    Rect trackWindow_;
    

    // COLOR DETECTION PARAMETERS
    /// Minimum value for the detection
    int min_sat_=100;
    /// Minimum saturation for the detection
    int min_value_=30;
    /// Sort type for color blob detection
    int sort_type_ = 3;
    /// List of all used colors
    vector <string> color_list_ = {"DarkBlue","Red","Orange","Yellow","Green","Blue","White","Purple"};
    /// Map of all used colors
    map < string, vector<cv::Scalar> > color_map_{
        {"DarkBlue",    {Scalar( 100,  min_sat_, min_value_), Scalar( 125, 255, 255)}},
        {"Red",         {Scalar(   0,  min_sat_, min_value_), Scalar(  4, 255, 255)}},
        {"DarkRed",     {Scalar( 170,  min_sat_, min_value_), Scalar( 180, 255, 255)}},
        {"Orange",      {Scalar(  4,  min_sat_, min_value_), Scalar(  30, 255, 255)}},
        {"Yellow",      {Scalar(  30,  min_sat_, min_value_), Scalar(  40, 255, 255)}},
        {"LimeGreen",   {Scalar(  40,  min_sat_, min_value_), Scalar(  60, 255, 255)}},
        {"Green",       {Scalar(  50,  min_sat_, min_value_), Scalar(  90, 255, 255)}},
        {"Blue",        {Scalar(  90,  min_sat_, min_value_), Scalar( 100, 255, 255)}},
        {"White",       {Scalar(   0,         0,        255), Scalar( 255,   0, 255)}},
        {"Pink",        {Scalar( 160,  min_sat_, min_value_), Scalar( 170, 255, 255)}},
        {"Purple",      {Scalar( 150,  min_sat_, min_value_), Scalar( 160, 255, 255)}},
        };
};
#endif
