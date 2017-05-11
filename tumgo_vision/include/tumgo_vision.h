#ifndef TUMGO_VISION
#define TUMGO_VISION

/** @file tumgo_vision.h
 *  @brief File storing declaration of the vision functions.
 */

// ROS library
#include <ros/ros.h>
#include <ros/duration.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

// Standard libraries
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <time.h>
#include <mutex>
#include <thread>
#include <pthread.h>
#include <vector>
#include <algorithm>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>

// Helper functions
#include <helpers.h>

// Services
#include <tumgo_vision/srvDetectColorBlob.h>
#include <tumgo_vision/srvDetectFace.h>
#include <tumgo_vision/srvDetectObject3D.h>
#include <tumgo_vision/srvSegmentObjects.h>

#include <tumgo_vision/srvRecognition.h>
#include <tumgo_vision/srvRecognition2D.h>
#include <std_srvs/Empty.h>

#include <tumgo_vision/srvImage.h>
#include <tumgo_vision/srvCloud.h>
#include <tumgo_vision/srvGetImage.h>
#include <tumgo_vision/srvGetCloud.h>

// Messages
#include <tumgo_vision/msgDetectObject.h>

// PCL specific includes
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>
#include <pcl/features/fpfh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl_ros/point_cloud.h> // enable pcl publishing
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/centroid.h> //for computing centroid

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>
#include <tf/transform_listener.h>

//thread
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/date_time.hpp>
#include <boost/thread/locks.hpp>

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud< PointT > PointCloud;
typedef pcl::PointCloud< PointT >::Ptr PointCloudPtr;
typedef pcl::Normal NormalT;
typedef pcl::SHOT352 DescriptorT;

using namespace std;
using namespace cv;

RNG rng(12345);

/// Structure for storing 2D objects
struct Objects2D{
    std::vector<string> labels;
    std::vector<float> probabilities;
    std::vector<std::vector<int> > sizes;
    std::vector<std::vector<int> > positions;
    std::vector<int> x;
    std::vector<int> y;
    std::vector<int> width;
    std::vector<int> height;
};
/// Structure for storing 3D objects
struct Objects3D{
    std::vector<string> labels;
    std::vector<string> colors;
    std::vector<float> probabilities;
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    std::vector<float> width;
    std::vector<float> height;
    std::vector<float> depth;
};

/// Vision class storing all services, messages and synchronizing image and pointcloud topics
class TUMgoVision{
public:
    /* *******************
     * Constructors
     * *******************/
    /// Constructor
    TUMgoVision(ros::NodeHandle n);
    /// Deconstructor
    ~TUMgoVision(){}

private:    
    /* *******************
     * Member functions
     * *******************/

    PointT convert2D(cv::Point2d req);
    
    cv::Point2d convert3D(PointT req);

    Objects3D compareAndCombine(Objects2D rec2D, Objects3D rec3D);

    Objects3D recognizeObjects();

    /* *******************
     * Service functions
     * *******************/
    bool getImageSRV(tumgo_vision::srvGetImage::Request  &req, tumgo_vision::srvGetImage::Response &res);

    bool getCloudSRV(tumgo_vision::srvGetCloud::Request  &req, tumgo_vision::srvGetCloud::Response &res);

    bool recognitionSRV(tumgo_vision::srvRecognition::Request  &req, tumgo_vision::srvRecognition::Response &res);

    /* *******************
     * Callback functions
     * *******************/
    /**
     * @brief for
     * @param msg: the plane-segmented point cloud
     */
    void processCloud(const sensor_msgs::PointCloud2::ConstPtr &msg);

    /**
     * @brief for storing the data in local variants
     * @param msg: the raw 2d image from the camera of robot
     */
    void imageCB(const sensor_msgs::ImageConstPtr &msg);

    void cloudCB(const sensor_msgs::PointCloud2::ConstPtr &cloud);

    void cameraCB(const sensor_msgs::CameraInfoConstPtr &camera);

    /**
     * @brief Synchronizer for pointcloud and image topics
     * @param cloud     Current point cloud from the robot, sensor_msgs::PointCloud2 
     * @param image     Current frame from the robot, sensor_msgs::Image
     */
    void synchronizeCB(const sensor_msgs::CameraInfoConstPtr &camera, const sensor_msgs::ImageConstPtr &image, const sensor_msgs::PointCloud2::ConstPtr &cloud);

    /// The node handle
    ros::NodeHandle nh_;
    /// Node handle in the private namespace
    ros::NodeHandle priv_nh_;

    //for saving the centroid position, width and height of the object
    Eigen::Vector4f centroid_obj_;
    float width_;
    float height_;

    // Services
    /// Detect 2D Object service client
    ros::ServiceClient recognition2D_client_;
    /// Detect 3D Object service client
    ros::ServiceClient detect_object3D_client_;
    /// Detect Color Blob service client
    ros::ServiceClient detect_color_client_;
    /// Detect Face service client
    ros::ServiceClient detect_face_client_;
    /// Update Image service client
    ros::ServiceClient update_image_client_;
    /// Update Cloud service client
    ros::ServiceClient update_cloud_client_;
    /// Segment Plane service client
    ros::ServiceClient segment_plane_client_;
    /// Segment Objects service client
    ros::ServiceClient segment_objects_client_;

    /// Get Image service server
    ros::ServiceServer get_image_service_;
    /// Get Cloud service server
    ros::ServiceServer get_cloud_service_;
    /// Recognition service server
    ros::ServiceServer recognition_service_;

    // Publishers
    /// Update Image publisher
    ros::Publisher update_image_pub_;
    /// Update Cloud publisher
    ros::Publisher update_cloud_pub_;

    // Subscribers
    /// PointCloud subscriber
    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    /// Camera frame subscriber
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    /// Camera parameters subscriber
    message_filters::Subscriber<sensor_msgs::CameraInfo> camera_sub_;
    /// Camera and PointCloud topics synchronizer
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CameraInfo, sensor_msgs::Image, sensor_msgs::PointCloud2> Approx1;
    message_filters::Synchronizer<Approx1> sync1;
    /// Camera Model
    image_geometry::PinholeCameraModel camera_model_;

    /// Image topic subscriber
    ros::Subscriber image_ns_sub_;
    /// PointCloud topic subscriber
    ros::Subscriber cloud_ns_sub_;
    /// Camera Info topic subscriber
    ros::Subscriber camera_ns_sub_;


    /// A tf transform listener
    tf::TransformListener listener_;

    /// Current frame from the camera
    cv::Mat curr_image_;
    /// Current pointcloud from the camera
    PointCloud curr_cloud_;
    /// Current frame from the camera in sensor_msgs format
    sensor_msgs::Image curr_cv_;
    /// Current pointcloud from the camera in sensor_msgs format
    sensor_msgs::PointCloud2 curr_pcl_;
    /// Current list of recognized objects
    Objects3D curr_objects_;

    /// Lock for image
    pthread_mutex_t image_mutex_ = PTHREAD_MUTEX_INITIALIZER;
    /// Lock for point cloud
    pthread_mutex_t cloud_mutex_ = PTHREAD_MUTEX_INITIALIZER;
    /// Flag for processing the image
    bool process_image_ = false;
    /// Flag for processing the cloud
    bool process_cloud_ = false;

    /// Parameter for showing raw image and pointcloud
    bool show_raw_;
    /// Parameter for showing processed images
    bool show_images_;
    /// Parameter for synchronizing point cloud with the image
    bool synchronize_;
    /// Parameter for stating whether we use kinect or robot
    bool robot_;   

    /// Parameter for the distance threshold for matching
    float dist_threshold_;
    /// Parameter for the probability threshold for matching
    float prob_threshold_;

};
#endif
