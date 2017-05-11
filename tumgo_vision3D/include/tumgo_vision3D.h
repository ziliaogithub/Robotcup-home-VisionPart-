#ifndef TUMGO_VISION3D
#define TUMGO_VISION3D

/** @file tumgo_vision3D.h
 *  @brief File storing declaration of the vision 3D functions.
 */

// ROS library
#include <ros/ros.h>
#include <ros/duration.h>
#include <tf/transform_listener.h>


// PCL specific includes
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h> // enable pcl publishing
#include <pcl_ros/transforms.h>
#include <pcl/correspondence.h>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h> //for computing centroid
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/pca.h>
#include <image_geometry/pinhole_camera_model.h>

// Standard libraries
#include <string>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <time.h>
//#include <mutex>
//#include <thread>
//#include <pthread.h>
#include <vector>
#include <algorithm>
#include <map>

//thread
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/date_time.hpp>
#include <boost/thread/locks.hpp>

// Helper functions
//#include <helpers.h>

// Services
#include <tumgo_vision/srvDetectObject3D.h>
#include <tumgo_vision/srvCloud.h>
#include <tumgo_vision/srvGetCloud.h>
#include <tumgo_vision/srvSegmentObjects.h>
#include <tumgo_vision/srvSegmentPlanes.h>
#include <tumgo_vision/srvRecognition3D.h>
#include <tumgo_vision/srvGetFacePosition.h>
#include <tumgo_vision/srvDetectFace.h>
#include <std_srvs/Empty.h>

// Messages
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseArray.h>


typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud< PointT > PointCloud;
typedef pcl::PointCloud< PointT >::Ptr PointCloudPtr;
typedef std::vector<PointCloudPtr, Eigen::aligned_allocator<PointCloud> >  PointCloudPtrVec;
typedef pcl::Normal NormalT;
typedef pcl::FPFHSignature33 DescriptorT;

//Algorithm params
bool rec_obj(false);
int inlier_threshold();// Min number of inliers for reliable plane detection
double down_sampling_voxel_size(0.01f);// Size of downsampling grid before performing plane detection
int sac_model_max_iterations(100);// Number of iterations to fit the consensus model
double sac_model_distance_threshold(0.02);// Distance for using the points in the consensus model
bool load_model(false);

PointCloudPtr off_scene_model (new PointCloud);
PointCloudPtr model (new PointCloud);

using namespace std;

typedef struct{
    string label;
    //Eigen::Vector4f centroid;
    float width;
    float depth;
    float height;
    float x;
    float y;
    float z;
    float confidence;
    PointCloud obj_pc;
    PointT min;
    PointT max;
}ObjectType;

//! Callback for processing the Point Cloud data

class TUMgoVision3D{
public:
    // Constructors
    TUMgoVision3D(ros::NodeHandle nh, std::string processing_frame, tf::TransformListener *listener);
    ~TUMgoVision3D(){};

    PointT convertPoint(PointT req);
    void showCloud(PointCloudPtr plane, PointCloudPtr objects);
    void showCloud(PointCloudPtr plane, PointCloudPtr objects, std::vector<PointT> all_objects);


    /* *******************
     * Local functions
     * *******************/
    /**
     * @brief Function for segmenting out the plane and extract different cluster using euclidean distance as metric
     * @param cloud_cluster: extracted point cloud clusters
     * @return True if succedeed, false otherwise.
     */
    bool segmentCloud(PointCloudPtrVec &cloud_cluster);

    /**
     * @brief Function for detecting specified object
     * @param cloud_cluster: extracted point cloud clusters from segmentCloud function
     * @return True if succedeed, false otherwise.
     */
    bool detectObject(PointCloudPtrVec &cloud_cluster);

    /**
     * @brief Function processing the found object in order to get its size and centroid position.
     * @param object_pc: the point cloud of object we want to process
     * @param label: the label of this object
     * @processing results will be stored in the class private parameters vector obj_list_
     */
    void processObject( PointCloud object_pc, 
                        float &width, 
                        float &depth, 
                        float &height, 
                        float &x, 
                        float &y, 
                        float &z, 
                        PointT &min, 
                        PointT &max);

    pcl::PointCloud<PointT>::Ptr segmentTable(pcl::PointCloud<PointT>::Ptr input, float per);
    string shapeDetector(pcl::PointCloud<PointT>::Ptr input);

    /* *******************
     * Service Callback functions
     * *******************/

    /**
     * @brief detectObject: processing the point cloud and reconigze the obj
     * @param req.label(string): the name of object we want to detect
     * @param res.result(bool):true if we detect successfully, else false
     * @param res.x,res.y,res.z(int32): the 3D postion of centroid of object based on the camera coordinate
     * @param res.width,res.height(int32): the size of object
     * @return res.result
     */
    bool detectObjectSRV(tumgo_vision::srvDetectObject3D::Request  &req, tumgo_vision::srvDetectObject3D::Response &res);

    /**
     * @brief getFacePosition: call detectFaceSrv in Vision2D and then extract the corresponding PointCloud and compute its centroid
     * @param req.personID: the name of person's face we want to detect
     * @param res.result(bool):true if we detect successfully, else false
     * @param res.x,res.y,res.z(int32): the 3D postion of centroid of object based on base_footprint frame
     * @param cloud(sensor_msgs/PointCloud2): extracted point cloud of face based on base_footprint frame
     * @return res.result
     */
    bool getFacePositionSRV(tumgo_vision::srvGetFacePosition::Request &req, tumgo_vision::srvGetFacePosition::Response &res);


    bool segmentPlaneSRV(tumgo_vision::srvSegmentPlanes::Request  &req, tumgo_vision::srvSegmentPlanes::Response &res);

    bool segmentObjectsSRV(tumgo_vision::srvSegmentObjects::Request  &req, tumgo_vision::srvSegmentObjects::Response &res);

    bool recognizeObjectsSRV(tumgo_vision::srvRecognition3D::Request  &req, tumgo_vision::srvRecognition3D::Response &res);


    /**
     * @brief Updating current pointcloud stored in curr_cloud .
     * @param req.cloud point cloud that should be stored
     * @return True if update successful, false otherwise.
     */
    bool updateCloudSRV(tumgo_vision::srvCloud::Request  &req, tumgo_vision::srvCloud::Response &res);
    
    /* *******************
     * Subscriber Callback functions
     * *******************/
    
    /**
     * @brief Updating current pointcloud stored in curr_cloud .
     * @param req.cloud     pointcloud that should be stored
     */
    void updateCloudCB(const sensor_msgs::PointCloud2::ConstPtr &msg);

private:
    /// The node handle
    ros::NodeHandle nh_;
    /// Node handle in the private namespace
    ros::NodeHandle priv_nh_;

    /// Parameter for showing raw image and pointcloud
    bool show_cloud_;
    /// Parameter for synchronizing point cloud with the image
    bool synchronize_;   
    /// Parameter for stating whether we use kinect or robot
    bool robot_;  

    //for saving the chracteristic of the object
    std::map<std::string, PointCloud> model_list_;
    std::vector<ObjectType> object_list_;

    // Services
    ros::ServiceServer detect_object_service_;
    ros::ServiceServer update_cloud_service_;
    ros::ServiceServer segment_plane_service_;
    ros::ServiceServer segment_objects_service_;
    ros::ServiceServer recognition_service_;
    ros::ServiceServer get_face_position_service_;

    //Clients
    ros::ServiceClient detect_face_client_;
    // Subscribers
    ros::Subscriber update_cloud_sub_;

    //! Publisher for segmented pointclouds
    ros::Publisher plane_pub_;
    ros::Publisher clusters_pub_;
    ros::Publisher object_pose_pub_;

    //! Publisher for segmented pointclouds
    ros::Publisher clusters_ec_pub_;

    /// Current pointcloud from the camera
    PointCloud curr_cloud_;    
    /// Current pointcloud from the camera
    sensor_msgs::PointCloud2 curr_pcl_;
    /// Current pointcloud of the table planes
    PointCloudPtr curr_table_;
    /// Current pointcloud of the cluster planes
    PointCloudPtr curr_object_;

    // EMILKA
    /// Current pointcloud of biggest plane
    PointCloudPtr curr_plane_;    
    /// Current pointcloud of the table
    PointCloudPtr curr_tables_;
    /// Current pointcloud of the objects
    PointCloudPtr curr_objects_;

    /// Current pointcloud of the objects
    std::vector<PointCloudPtr> all_objects_;
    /// Current min/max points of the objects
    std::vector<PointT> all_points_;

    ///the pose of found object
    geometry_msgs::PoseArray curr_object_poses_;

    /// Lock for point cloud
    //pthread_mutex_t count_mutex_ = PTHREAD_MUTEX_INITIALIZER;
    /// Flag for processing the cloud
    bool process_cloud_;

    /// Clouds are transformed into this frame before processing; leave empty if clouds are to be processed in their original frame
    std::string processing_frame_;
    string model_file;

    /// A tf transform listener
    tf::TransformListener *listener_;

    //! Parameters
    float tolerance; 
    int min_size;
    int max_size; 
    float threshold; 
    float percent; 
    std::string models_dir;
};
#endif
