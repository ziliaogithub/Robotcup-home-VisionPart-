/* ***************************************
    Author: Emilia Lozinska
    E-mail: e.lozinska@tum.de
    Author: Jianxiang Feng
    E-mail: jianxiang.feng@tum.de
*/

#include <tumgo_vision.h>

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool pc_new(false);
bool rec_obj(false);
bool thread(false);
float model_ss_ (0.01f);
float scene_ss_ (0.03f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);
int moved = 0;

PointCloudPtr off_scene_model (new PointCloud);
PointCloudPtr model (new PointCloud);

float distance(PointT x1, PointT x2)
{
    return sqrt(pow((x1.x-x2.x),2)+pow((x1.y-x2.y),2)+pow((x1.z-x2.z),2));
}

/* *******************
 * Member functions
 * *******************/
PointT TUMgoVision::convert2D(cv::Point2d req)
{
    PointT p3d = curr_cloud_(req.x,req.y);
    // put data into the response
    if ( (p3d.x == p3d.x) && (p3d.y == p3d.y) && (p3d.z == p3d.z) ) // check NaN
        return p3d;
    else
        return PointT();
}

cv::Point2d TUMgoVision::convert3D(PointT req)
{
    // geometry_msgs::PointStamped point,point2;
    // point.header.frame_id = "base_footprint";
    // point.header.stamp = ros::Time::now();
    // point.point.x = req.x;
    // point.point.y = req.y;
    // point.point.z = req.z;
    // listener_.waitForTransform("/base_footprint", "/xtion_rgb_optical_frame",
    //                           ros::Time::now(), ros::Duration(5.0));
    // listener_.transformPoint("xtion_optical_frame", point, point2);

    // // Create the 3D OpenCV point
    // cv::Point3d point_cv(point2.point.x, point2.point.y, point2.point.z);
    cv::Point3d point_cv(req.x, req.y, req.z);
    // Create the 2D OpenCV point
    cv::Point2d point_image;
    // Convert between 3D and 2D
    point_image = camera_model_.project3dToPixel(point_cv);
    cout << "CV point: " << point_image.x << " " << point_image.y << endl;

    // Return the result
    return point_image;
}

Objects3D TUMgoVision::compareAndCombine(Objects2D rec2D, Objects3D rec3D)
{
    Objects3D result;
    // iterate through all 2D recognitions
    for (int i = 0; i < rec2D.sizes.size(); i++)
    {
        // convert to 3D point
        PointT center_2D = convert2D(cv::Point(rec2D.positions[i][0],rec2D.positions[i][1]));
        float dist = -10;
        float new_dist;
        int index = 0;
        for (int j = 0; j < rec3D.labels.size(); j++)
        {
            PointT center_3D;
            center_3D.x = rec3D.x[j];
            center_3D.y = rec3D.y[j];
            center_3D.z = rec3D.z[j];
            new_dist = distance(center_2D, center_3D);
            if (dist < new_dist)
            {
                index = j;
                dist = new_dist;
            }
        }
        // check the probability
        if (rec2D.probabilities[i]+rec3D.probabilities[index] < prob_threshold_)
        {
            // Push back to combinations
            result.labels.push_back(rec2D.labels[i]);
            result.probabilities.push_back(rec2D.probabilities[i]+rec3D.probabilities[index]);
            result.x.push_back(rec3D.x[index]);
            result.y.push_back(rec3D.y[index]);
            result.z.push_back(rec3D.z[index]);
            result.width.push_back(rec3D.width[index]);
            result.height.push_back(rec3D.height[index]);
            result.depth.push_back(rec3D.depth[index]);
        }
        // check the distance
        if (dist < dist_threshold_)
        {
            // Erase from the 3D vector
            rec3D.labels.erase(rec3D.labels.begin() + index);
            rec3D.probabilities.erase(rec3D.probabilities.begin() + index);
            rec3D.x.erase(rec3D.x.begin() + index);
            rec3D.y.erase(rec3D.y.begin() + index);
            rec3D.z.erase(rec3D.z.begin() + index);
            rec3D.width.erase(rec3D.width.begin() + index);
            rec3D.height.erase(rec3D.height.begin() + index);
            rec3D.depth.erase(rec3D.depth.begin() + index);
        }
    }
    return result;
}

// Objects3D TUMgoVision::recognizeObjects()
// {
//     // Declare local variables
//     Objects2D recognition_2D;
//     Objects3D recognition_3D;
//     Objects3D result;
//     // Do the 2D recognition
//     tumgo_vision::srvRecognition2D obj_2D;
//     obj_2D.request.frame = curr_cv_;
//     //do 3D object detection
//     tumgo_vision::srvDetectObject3D obj_3D;

//     // STEP 1: Recognize objects in 2D image
//     ROS_INFO("Step 1");
//     if(recognition2D_client_.call(obj_2D))
//     {
//         // Insert response into the structure
//         recognition_2D.labels = obj_2D.response.labels;
//         recognition_2D.probabilities = obj_2D.response.percent;
//         for (int i = 0; i < obj_2D.response.bb_minx.size(); i++)
//         {
//             recognition_2D.positions.push_back({(obj_2D.response.bb_maxx[i]-obj_2D.response.bb_minx[i])/2,
//                 (obj_2D.response.bb_maxy[i]-obj_2D.response.bb_miny[i])/2});
//             recognition_2D.sizes.push_back({abs(obj_2D.response.bb_maxx[i]-obj_2D.response.bb_minx[i]),
//                 abs(obj_2D.response.bb_maxy[i]-obj_2D.response.bb_miny[i])});
//         }
//         // STEP 2: Recognition in 3D
//         if(detect_object3D_client_.call(obj_3D))
//         {
//             if(obj_3D.response.result)
//             {
//                 for(unsigned i = 0; i<obj_3D.response.label.size(); ++i)
//                 {
//                     recognition_3D.labels.push_back(obj_3D.response.label[i]);
//                     recognition_3D.probabilities.push_back(obj_3D.response.confidence[i]);
//                     recognition_3D.x.push_back(obj_3D.response.x[i]);
//                     recognition_3D.y.push_back(obj_3D.response.y[i]);
//                     recognition_3D.z.push_back(obj_3D.response.z[i]);
//                     recognition_3D.width.push_back(obj_3D.response.width[i]);
//                     recognition_3D.height.push_back(obj_3D.response.height[i]);
//                     recognition_3D.depth.push_back(obj_3D.response.depth[i]);
//                 }
//             }
//             else  std::cout<<"no object in point cloud was detected!"<<"\n"<<std::endl;
//         }
//         else std::cout<<"calling 3d object detection service failed!"<<"\n"<<std::endl;
//         // STEP 3: Combining both of them
//         result = compareAndCombine(recognition_2D, recognition_3D);
//     }
//     else
//     {
//         ROS_ERROR("Couldn't perform recognition in 2D.");
//     }
//     return result;
// }

Objects3D TUMgoVision::recognizeObjects()
{
    // Declare local variables
    Objects2D recognition_2D;
    Objects3D recognition_3D;
    Objects3D result;

    // Do the 2D recognition
    tumgo_vision::srvRecognition2D obj_2D;
    obj_2D.request.frame = curr_cv_;
    //do 3D object detection
    tumgo_vision::srvDetectObject3D obj_3D;

    // STEP 1: Recognition in 3D
    if(detect_object3D_client_.call(obj_3D))
    {
        if(obj_3D.response.result)
        {
            for(unsigned i = 0; i<obj_3D.response.label.size(); ++i)
            {
                recognition_3D.labels.push_back(obj_3D.response.label[i]);
                recognition_3D.probabilities.push_back(obj_3D.response.confidence[i]);
                recognition_3D.x.push_back(obj_3D.response.x[i]);
                recognition_3D.y.push_back(obj_3D.response.y[i]);
                recognition_3D.z.push_back(obj_3D.response.z[i]);
                recognition_3D.width.push_back(obj_3D.response.width[i]);
                recognition_3D.height.push_back(obj_3D.response.height[i]);
                recognition_3D.depth.push_back(obj_3D.response.depth[i]);
                // Convert the points and check color
                PointT min,max;
                min.x = obj_3D.response.min_x[i];
                min.y = obj_3D.response.min_y[i];
                min.z = obj_3D.response.min_z[i];
                max.x = obj_3D.response.max_x[i];
                max.y = obj_3D.response.max_y[i];
                max.z = obj_3D.response.max_z[i];
                cv::Point2d p_min = convert3D(min);
                cv::Point2d p_max = convert3D(max);
                tumgo_vision::srvDetectColorBlob color;
                color.request.bbox = true;
                color.request.in_x = (p_min.x < 0 ? 0 : p_min.x) + abs(p_max.x - p_min.x)*0.1; 
                color.request.in_y = (p_min.y < 0 ? 0 : p_min.y) + abs(p_max.x - p_min.x)*0.1;
                color.request.in_width = (p_max.x  < curr_image_.rows ? abs(p_max.x - p_min.x) : curr_image_.rows -p_min.x)*0.8;
                color.request.in_height = (p_max.y  < curr_image_.cols  ? abs(p_max.y - p_min.y) : curr_image_.cols -p_min.y)*0.8;
                if (abs(p_max.x - p_min.x) > 5) 
                {
                    if(detect_color_client_.call(color))
                        recognition_3D.colors.push_back(color.response.color);
                    else
                    {
                        ROS_ERROR("Failed to call color detection");
                        recognition_3D.colors.push_back("Unknown");
                    }
                }
                else
                    recognition_3D.colors.push_back("Unknown");
            }
        }
        else  std::cout<<"no object in point cloud was detected!"<<"\n"<<std::endl;
    }
    else std::cout<<"calling 3d object detection service failed!"<<"\n"<<std::endl;
        // STEP 3: Combining both of them
    return recognition_3D;
}
/* *******************
 * Service functions
 * *******************/
bool TUMgoVision::getImageSRV(tumgo_vision::srvGetImage::Request  &req, tumgo_vision::srvGetImage::Response &res)
{
    // If image is not processed at the moment
    if(!process_image_)
    {
        res.frame = curr_cv_;
        res.valid = true;
    }
    else
        res.valid = false;
    return true;
}

bool TUMgoVision::getCloudSRV(tumgo_vision::srvGetCloud::Request  &req, tumgo_vision::srvGetCloud::Response &res)
{
    // If cloud is not processed at the moment
    if(!process_cloud_)
    {
        res.cloud = curr_pcl_;
        res.valid = true;
    }
    else
        res.valid = false;
    return true;
}

bool TUMgoVision::recognitionSRV(tumgo_vision::srvRecognition::Request  &req, tumgo_vision::srvRecognition::Response &res)
{
    // Lock the resources
    ROS_INFO("Recognition");
    pthread_mutex_lock( &this->image_mutex_ );
    process_image_ = true;
    process_cloud_ = true;
    // If required, do the recognition again
    if (req.type == 0) 
    {
        ROS_INFO("Recognition started");
        curr_objects_ = recognizeObjects();
    }
    if(!curr_objects_.labels.empty())
    {
        for(unsigned i = 0; i<curr_objects_.labels.size(); i++)
        {
            res.shapes.push_back(curr_objects_.labels[i]);
            res.colors.push_back(curr_objects_.colors[i]);
            res.percents.push_back(curr_objects_.probabilities[i]);
            res.x.push_back(curr_objects_.x[i]);
            res.y.push_back(curr_objects_.y[i]);
            res.z.push_back(curr_objects_.z[i]);
            res.width.push_back(curr_objects_.width[i]);
            res.height.push_back(curr_objects_.height[i]);
            res.depth.push_back(curr_objects_.depth[i]);
            res.result = true;
        }
    }
    else res.result = false;
    process_image_ = false;
    process_cloud_ = false;
    pthread_mutex_unlock( &this->image_mutex_ );
    return true;
}

/* *******************
 * Callback functions
 * *******************/
void TUMgoVision::synchronizeCB(const sensor_msgs::CameraInfoConstPtr &camera, const sensor_msgs::ImageConstPtr &image, const sensor_msgs::PointCloud2::ConstPtr &cloud)
{
    // Step 1: Get the images and store locally
    // Reduce the frame rate
    pcl::PointCloud< pcl::PointXYZ > pc;
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat rgb;
    try
    {
        cv_ptr = cv_bridge::toCvCopy( image, sensor_msgs::image_encodings::BGR8 );
        curr_image_ = cv_ptr->image;
        curr_cv_ = *image;
        pcl::fromROSMsg(*cloud, curr_cloud_);
        curr_pcl_ = *cloud;
        camera_model_.fromCameraInfo(camera);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    // Step 2: Update the images in other nodes
    tumgo_vision::srvImage srv_image;
    tumgo_vision::srvCloud srv_cloud;
    srv_image.request.frame = curr_cv_;
    srv_cloud.request.cloud = curr_pcl_;
    if (!update_image_client_.call(srv_image))
        ROS_ERROR("Couldn't update the frame");
    if (!update_cloud_client_.call(srv_cloud))
        ROS_ERROR("Couldn't update the cloud");
}

void TUMgoVision::imageCB(const sensor_msgs::ImageConstPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    if(!process_image_)
    {
        curr_image_ = cv_ptr->image;
        curr_cv_ = *msg;
    }
    if (show_images_)
    {
        imshow("RAW IMAGE",curr_image_);
        waitKey(1);
    }
}

void TUMgoVision::cloudCB(const sensor_msgs::PointCloud2::ConstPtr &cloud)
{
    pcl::fromROSMsg(*cloud, curr_cloud_);
    curr_pcl_ = *cloud;
}

void TUMgoVision::cameraCB(const sensor_msgs::CameraInfoConstPtr &camera)
{
    camera_model_.fromCameraInfo(camera);
}

// Constructor
TUMgoVision::TUMgoVision(ros::NodeHandle nh) : nh_(nh), priv_nh_("~"),
    image_sub_(nh, "/xtion/rgb/image_raw", 20), cloud_sub_(nh, "/xtion/depth_registered/points", 20),
    camera_sub_(nh, "/xtion/rgb/camera_info",20),
    sync1(Approx1(20), camera_sub_, image_sub_, cloud_sub_)
{
    nh_.param<bool>("synchronize", synchronize_, false);
    nh_.param<bool>("show_images", show_images_, true);
    nh_.param<bool>("robot", robot_, true);

    detect_color_client_ = nh_.serviceClient<tumgo_vision::srvDetectColorBlob>("tumgo_vision/detect_color");
    recognition2D_client_ = nh_.serviceClient<tumgo_vision::srvRecognition2D>("tumgo_vision/recognition2D");
    detect_object3D_client_ = nh_.serviceClient<tumgo_vision::srvDetectObject3D>("tumgo_vision/detect_object3D");
    update_image_client_ = nh_.serviceClient<tumgo_vision::srvImage>("tumgo_vision/update_image");
    update_cloud_client_ = nh_.serviceClient<tumgo_vision::srvCloud>("tumgo_vision/update_cloud");

    get_image_service_ = nh_.advertiseService("/tumgo_vision/get_image",&TUMgoVision::getImageSRV,this);
    get_cloud_service_ = nh_.advertiseService("/tumgo_vision/get_cloud",&TUMgoVision::getCloudSRV,this);
    recognition_service_ = nh_.advertiseService("/tumgo_vision/recognition",&TUMgoVision::recognitionSRV,this);

    // Publishers
    update_image_pub_ = nh_.advertise<sensor_msgs::Image>("tumgo_vision/update_image", 1000);
    update_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("tumgo_vision/update_cloud", 1000);

    if (synchronize_)
        sync1.registerCallback(boost::bind(&TUMgoVision::synchronizeCB,this, _1, _2, _3));
    else
    {
        // Subscribe to input video feed and publish output video feed.
        if (robot_)
        {
           image_ns_sub_ = nh_.subscribe("/xtion/rgb/image_raw",20,&TUMgoVision::imageCB,this);
           cloud_ns_sub_ = nh_.subscribe("/xtion/depth_registered/points",20,&TUMgoVision::cloudCB,this);
           camera_ns_sub_ = nh_.subscribe("/xtion/rgb/camera_info",20,&TUMgoVision::cameraCB,this);
        }
        else
           image_ns_sub_ = nh_.subscribe("/kinect2/qhd/image_color_rect",1,&TUMgoVision::imageCB,this);
    }
    dist_threshold_ = 0.01;
    prob_threshold_ = 0.8;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tumgo_vision");
    ros::NodeHandle n;
    TUMgoVision node(n);
    ROS_INFO("Started TUMgo_vision node.");
    ros::spin();
    return 0;
}


