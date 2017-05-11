#include <tumgo_human_detection.h>

// Initialize static members.
bool TUMgoHumanDetection::m_newImage_static = false;
bool TUMgoHumanDetection::m_newBB_static = false;

/**
 * @brief      Constructor.
 */
TUMgoHumanDetection::TUMgoHumanDetection(ros::NodeHandle nh)
    : m_node(nh)
    , m_it(nh)
{
    m_node.param<bool>("show_raw", show_raw_, true);
    m_node.param<bool>("show_images", show_images_, false);
    m_node.param<bool>("synchronize", synchronize_, true);
    m_node.param<bool>("robot", robot_, true);
    nh.param<std::string>("cascades", cascades, "/home/rcah/ros/workspaces/project_ws/src/tumgo_vision/cascades");

    // Member variable initialization.
    std::string m_windowName = "Human body detection";
    m_queuesize = 2;

    
    // Subscribers
    if (synchronize_)
        update_image_sub_ = m_node.subscribe("/tumgo_vision/update_image",1,&TUMgoHumanDetection::updateImageCB,this);
    else
    {
        // Subscribe to input video feed and publish output video feed.
        if (robot_)
           update_image_sub_ = m_node.subscribe("/xtion/rgb/image_raw",1,&TUMgoHumanDetection::updateImageCB,this);
        else
           update_image_sub_ = m_node.subscribe("/kinect2/qhd/image_color_rect",1,&TUMgoHumanDetection::updateImageCB,this);
    }

    // Service servers
    detect_human_service = nh.advertiseService("/tumgo_vision/detect_human",&TUMgoHumanDetection::detectHumanSRV,this);
}

/**
 * @brief      Destroys the object.
 */
TUMgoHumanDetection::~TUMgoHumanDetection()
{}

/**
 * @brief      Function for detecting and displaying the human bodies.
 *
 * @param[in]  frame  The frame in which to search for the body.
 */
void TUMgoHumanDetection::detectAndDisplay(std::string cascade_file)
{
    std::string cascade_name = cascades + cascade_file;

    // To gray image.
    cv::Mat frame_gray;

    // // TESTING
    // cv::cvtColor(curr_image_, frame_gray, CV_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    cv::CascadeClassifier cascade;

    // Load the cascades.
    //-- 1. Load the cascades
    if (!cascade.load(cascade_name))

    {
        ROS_ERROR("Problem loading the cascade %s", cascade_name.c_str());
        return;
    }
    ROS_INFO("Cascade loaded.");
    // testing
    cv:Mat img = curr_image_;
    // cv::Mat img = frame_gray;

    // Found people.
    std::vector<cv::Rect> found, found_filtered;

    // HOG people detector.
    // cv::HOGDescriptor hog;
    // hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    // hog.detectMultiScale(img, found, 1.2, Size(8,8), Size(128,128), 1.2, 2); // 1.05

    // Haar detector.
    cascade.detectMultiScale(img, found, 1.2, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

    size_t i, j;

    for (i = 0; i < found.size(); i++)
    {
        cv::Rect r = found[i];

        for (j = 0; j < found.size(); j++)
        {
            if (j != i && (r & found[j]) == r)
            {
                break;
            }
        }

        if (j == found.size())
        {
            found_filtered.push_back(r);
        }
    }

    ROS_INFO("Found filtered.");
    for (i = 0; i < found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(img, r.tl(), r.br(), Scalar(0,255,0), 3);
    }

#ifdef DEBUG // Enable/Disable in the header.
    displayImage(img);
#endif

}


bool TUMgoHumanDetection::updateImageSRV(tumgo_vision::srvImage::Request  &req, tumgo_vision::srvImage::Response &res)
{
    // Reduce the frame rate
    ros::Duration(0.5).sleep();
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(req.frame, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        res.updated = false;
        return true;
    }

    Mat InImage=cv_ptr->image;
    if(!process_image_)
        curr_image_ = InImage;
    if (show_raw_)
    {
        imshow("Camera frame",InImage);
        waitKey(1);
    }
    res.updated = true;
    return true;
}

bool TUMgoHumanDetection::detectHumanSRV(tumgo_vision::srvDetectHuman::Request  &req, tumgo_vision::srvDetectHuman::Response &res)
{
    std::string cascade_file;

    if (req.toDetect == "body") {
        cascade_file = "/haar_upper_body_cascade.xml";
    } else if (req.toDetect == "face") {
        cascade_file = "/haarcascade_frontalface_alt.xml";
    } else if (req.toDetect == "profile") {
        cascade_file = "/haarcascade_profileface.xml";
    } else {
        ROS_ERROR("Could not find corresponding cascade file. Valid options: body, face, profile");
        res.result = false;
        return true;
    }

    TUMgoHumanDetection::detectAndDisplay(cascade_file);

    res.result = true;
    return true;
}

/* *******************
 * Callback functions
 * *******************/
void TUMgoHumanDetection::updateImageCB(const sensor_msgs::ImageConstPtr &msg)
{
    // Reduce the frame rate
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

    Mat InImage=cv_ptr->image;
    if(!process_image_)
        curr_image_ = InImage;
    if (show_raw_)
    {
        imshow("RAW IMAGE",InImage);
        //detectAndDisplay(curr_image_);
        waitKey(1);
    }
}


/* Private methods. */

/**
 * @brief      Display the image on the screen.
 */
void TUMgoHumanDetection::displayImage(Mat img)
{
    // Visualize the image with the frame.
    cv::imshow( m_windowName, img);
    cv::waitKey(3);
}
