/* ***************************************
    Author: Emilia Lozinska
    E-mail: e.lozinska@tum.de
    Author: Qiuhai Guo
    E-mail: qiuhai.guo@tum.de
*/
#include <tumgo_vision2D.h>
using namespace cv;

/* ************************ *
 * Preprocessing functions  *
 * ************************ */
cv::Mat TUMgoVision2D::clahe(cv::Mat img, float ClipLimit, int TilesGridSize)
{
    // normalize image contrast & luminance
    cv::Mat norm_img;
    cv::cvtColor(img, norm_img, CV_BGR2Lab);

    // Extract the L channel
    std::vector<Mat> lab_planes(3);
    cv::split(norm_img, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<CLAHE> clahe_obj = cv::createCLAHE();
    clahe_obj->setClipLimit(ClipLimit);
    clahe_obj->setTilesGridSize(Size(TilesGridSize,TilesGridSize));
    cv::Mat dst;
    clahe_obj->apply(lab_planes[0], dst);

    // Merge the the color planes back into an Lab image
    dst.copyTo(lab_planes[0]);
    cv::merge(lab_planes, norm_img);
    cv::cvtColor(norm_img, norm_img, CV_Lab2BGR);
    if (show_images_)
      cv::imshow("CLAHE", norm_img);
    return norm_img;
}

cv::Mat TUMgoVision2D::rotate(cv::Mat img,cv::Point2f center, double angle)
{
    cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1);
    cv::Mat result;
    cv::warpAffine(img, result, rot, img.size());
    if (show_images_)
        cv::imshow("Rotation", result);
    return result;
}

cv::Mat TUMgoVision2D::filterImage(cv::Mat img, int type)
{
    if (img.empty())
        return img;
    cv::Mat blurred;
    cv::Mat erodeElement = cv::getStructuringElement(2, cv::Size(3, 3), cv::Point(1, 1));
    cv::Mat dilateElement = cv::getStructuringElement(2, cv::Size(5, 5), cv::Point(3, 3));               
    switch(type){
        case 0:
            // Gaussian blur applied
            cv::GaussianBlur(img,blurred,cv::Size( 5, 5),0,0);
        break;
        case 1:
            // Median blur applied
            cv::medianBlur(img,blurred, 2);
        break;
        case 2:
            // Filter image without losing borders
            cv::bilateralFilter(img,blurred,5,10,2);
        break;
        case 3:
            // Adaptive bilateral filter
            cv::adaptiveBilateralFilter(img,blurred,cv::Size( 5, 5),10);
        break;        
        case 4:
            // Blurs an image using the normalized box filter
            cv::blur(img,blurred,cv::Size( 5, 5));
        break;
        case 5:
            // Erode
            cv::erode(img, blurred, erodeElement);
        break;
        case 6:
            // Dilate
            cv::dilate(img, blurred, dilateElement);
        break;
    }
    if (show_images_)
        cv::imshow("Filtered image", blurred);
    return blurred;
}

vector<cv::Point> TUMgoVision2D::extractColor(Mat img, string color)
{
    vector<cv::Point> result;
    if (img.empty())
        return result;
    img = clahe(img,2,6);
    img = filterImage(img,4);
    //  HSV image
    Mat hsvImage;
    //  Convert image to HSV
    cvtColor( img, hsvImage, CV_BGR2HSV);

    //  Extract color
    Mat color_img;
    Scalar color_min = this->color_map_[color][0];
    Scalar color_max = this->color_map_[color][1];
    inRange(hsvImage, color_min, color_max, color_img); // Extract the color
    // Dilate and erode
    color_img = filterImage(color_img,6);
    color_img = filterImage(color_img,5);

    string name = color;
    name += " color blob";
    // Show result
    if (show_colors_)
        imshow(name, color_img);
    //  Detecting Contour
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    RotatedRect boundRect;
    /// Find contours
    cv::findContours(color_img, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    if (!(contours.size() > 0))
        return result;
    else
    {
        // Compare wrt. the Y coordinate
        if(sort_type_ == 1){
            std::sort(contours.begin(),contours.end(),compare_y);
        }
        // Compare wrt. the X coordinate
        else if(sort_type_ == 2){
            std::sort(contours.begin(),contours.end(),compare_x);
        }
        // Compare wrt. the area of the rectangle bounding the blob
        else if(sort_type_ == 3){
            std::sort(contours.begin(),contours.end(),compare_area);
        }
        // Compare wrt. the area of the size of the contour
        else {
            std::sort(contours.begin(),contours.end(),compare_size);
        }

        // Reverse the sort to have the biggest values first
        reverse(contours.begin(),contours.end());

        //Save biggest blobs
        boundRect= minAreaRect(Mat(contours[0]));
        Point2f center = boundRect.center;

        Mat show = img.clone();
        // Used for RGB image show
        Scalar shadow_color = Scalar( 0, 255, 0);
        // Only if blob is detected, draw
        drawContours(show, contours, 0, shadow_color, 2, 8, hierarchy, 0, Point() );
        string name = color;
        name += " color extracted";
        // Show result
        if (show_colors_)
            imshow(name, show);
        waitKey(3);
    }
    // Take the middle of the two central points
    return contours[0];
}

bool TUMgoVision2D::getImage(void)
{
    if (this->process_image_)
        return false;
    else
    {
        tumgo_vision::srvGetImage srv;
        if(get_image_client_.call(srv))
        {    
            cv_bridge::CvImagePtr cv_ptr;
            try
            {
                cv_ptr = cv_bridge::toCvCopy(srv.response.frame, sensor_msgs::image_encodings::BGR8);
            }
            catch (cv_bridge::Exception& e)
            {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return false;
            }
            curr_image_ = cv_ptr->image;
            return true;
        }
        else
            return false;
    }
}

/* ****************** *
 * Service functions  *
 * ****************** */
bool TUMgoVision2D::detectColorBlobSRV(tumgo_vision::srvDetectColorBlob::Request  &req, tumgo_vision::srvDetectColorBlob::Response &res)
{
    if (this->curr_image_.empty())
        return false;
    // Lock the resources
    pthread_mutex_lock( &this->image_mutex_ );
    // Set the flag to true
    process_image_ = true;

    vector<cv::Point> temp, max_size;
    bool first_check = false;
    string max_arg = "None";
    for(std::map<string,vector<cv::Scalar> >::iterator iter = color_map_.begin(); iter != color_map_.end(); ++iter)
    {
        if (!req.bbox)
            temp = (extractColor(this->curr_image_,iter->first));
        else
        {
            Mat bbox_img;
            Mat mask = Mat::zeros(this->curr_image_.rows, this->curr_image_.cols, CV_8U);
            mask(Rect(req.in_x, req.in_y, req.in_width, req.in_height)) = 255;
            this->curr_image_.copyTo(bbox_img, mask);
            temp = (extractColor(bbox_img,iter->first));
        }
        if (!first_check) 
        {
            if (temp.size())
                first_check = true;
            max_size = temp;
            max_arg = iter->first;
        }
        else if (temp.size() && compare_area(max_size,temp))
        {
            max_size = temp;
            max_arg = iter->first;
        }
    }
    if (max_arg != "None")
    {
        RotatedRect boundRect = minAreaRect(Mat(max_size));
        float area = boundRect.size.width*boundRect.size.height;
        if (!req.bbox)
        {  
            if (area > this->curr_image_.rows * this->curr_image_.cols * 0.01)
                res.valid = true;
            else
                res.valid = false;
        }
        else
        {  
            if (area > req.in_width * req.in_height * 0.25)
                res.valid = true;
            else
                res.valid = false;
        }
        Point2f center = boundRect.center;
        res.color = max_arg;
        cout << max_arg << ", area:" << area;
        res.x = center.x;
        res.y = center.y;
        res.width = boundRect.size.width;
        res.height = boundRect.size.height;
    }
    // Unlock the resources
    pthread_mutex_unlock( &this->image_mutex_ );
    process_image_ = false;

    return true;
}

bool TUMgoVision2D::detectFaceSRV(tumgo_vision::srvDetectFace::Request  &req, tumgo_vision::srvDetectFace::Response &res)
{
    cv::Mat img,gray_img,faceDet;
    std::vector<Rect> facesRect;
    bool result=false;

    string face_cascade_name = cascades_pth_+"/haarcascade_frontalface_alt.xml";
    CascadeClassifier face_cascade;
    //string window_name = "Face recognition";

    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face\n"); return -1; };

    Ptr<FaceRecognizer> model1 = createEigenFaceRecognizer();
    string eigenfaces = cascades_pth_+"/eigenfaces_at.yml";
    model1->load(eigenfaces);

    img=curr_image_;
    cv::resize(img,img,cv::Size(),0.5,0.5);
    cv::cvtColor(img,gray_img, CV_BGR2GRAY);
    cout<<"detect face"<<endl;
    face_cascade.detectMultiScale( gray_img, facesRect, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    for( size_t i = 0; i < facesRect.size(); i++ )
    {
         gray_img(facesRect[i]).copyTo(faceDet);
         cv::resize(faceDet,faceDet,cv::Size(100,100));
         int predictedLabel=-1;
         double predicted_confidence = 0.0;
         model1->predict(faceDet,predictedLabel,predicted_confidence);
         string result_message = format("Predicted class = %d / predicted confidence = %f.", predictedLabel, predicted_confidence);
         cout << result_message << endl;
         // cout<<"predicted_confidence "<< predicted_confidence<<endl;
         if (predicted_confidence>6000)
            continue;
        result=true;
        res.personID = predictedLabel;
        res.x=facesRect[i].x;
        res.y=facesRect[i].y;
        res.width=facesRect[i].width;
        res.height=facesRect[i].height;
        break;
    }
    if (result)
        res.result=true;
    else
        res.result=false;
    return true;
}

bool TUMgoVision2D::startTracking(Rect trackWindow_){
    // Extract ROI in image
    Mat roi = this->curr_image_(trackWindow_);
    rectangle(this->curr_image_,trackWindow_,Scalar(0,255,0));
    // convert image to HSV
    Mat hsv_image;
    cvtColor( roi, hsv_image, CV_BGR2HSV);
    // Split channels
    Mat hue_image;
    Mat sat_image;
    std::vector<Mat> channels(3);
    
    split(hsv_image, channels);  
    hue_image = channels[0]; // Hue channel
    sat_image = channels[1]; // Saturation channel
    
    // Threshold of the saturation image
    Mat threshold_image;
    threshold(sat_image,threshold_image, 50, 255,  CV_THRESH_BINARY);
    // Mask hue image
    bitwise_and(hue_image,threshold_image,hue_image);
    // Calculate histogram
    calcHist(&hue_image,1,0,Mat(),hist_target_,1,&histSize_,&ranges_,true,false);
    normalize(hist_target_,hist_target_,0,255,32,-1,Mat());
    return true;
}

Rect TUMgoVision2D::tracking(Rect track_window)
{
    Mat hsv_img;
    // convert image to HSV
    cvtColor(this->curr_image_, hsv_img, CV_BGR2HSV);
    
    // Split channels
    Mat hue_image;
    Mat sat_image;
    std::vector<Mat> channels(3);
    
    split(hsv_img, channels);  
    hue_image = channels[0]; // Hue channel
    sat_image = channels[1]; // Saturation channel

    // Threshold of the saturation image
    Mat threshold_image;
    threshold(sat_image,threshold_image, 200, 255,  CV_THRESH_BINARY);
    // Mask hue image
    hue_image &= threshold_image;
    
    // Back projection
    MatND backproj;
    calcBackProject(&hue_image,1,0,hist_target_,backproj,&ranges_,1,true);
    
    Mat track_img;
    this->curr_image_.copyTo(track_img);
    
    // Mask backproj
    backproj &= threshold_image;
    
    if(mean_)
    {
        meanShift(backproj, trackWindow_,
            TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
        rectangle(track_img,trackWindow_,Scalar(0,255,0));
    }
    else
    {
        RotatedRect trackBox = CamShift(backproj, trackWindow_,
            TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
        
        if (trackWindow_.area() <= 1){ // reset target window if too small, otherwise segfault
            trackWindow_ = startTrackWindow_;
        }
        Point2f vertices[4];
        trackBox.points(vertices);
        for (int i = 0; i < 4; i++)
        line(track_img, vertices[i], vertices[(i+1)%4], Scalar(0,255,0));
    }
    if (show_images_)    
        imshow("Tracked object",track_img);
    return trackWindow_;
}

bool TUMgoVision2D::trackObjectSRV(tumgo_vision::srvTrackObject::Request  &req, tumgo_vision::srvTrackObject::Response &res)
{
    Mat hsv_img;
    if(req.start)
    {
        trackWindow_ = Rect(req.x - req.width/2, req.y - req.height/2, req.width, req.height);
        startTrackWindow_ = trackWindow_;
        startTracking(trackWindow_);
        tracking_ = true;
    }
    if(req.stop)
    {
        tracking_ = false;
    }

    Point2f center = Point((trackWindow_.x + trackWindow_.width)/2, (trackWindow_.y + trackWindow_.height)/2);
    res.x = center.x;
    res.y = center.y;
    res.width = trackWindow_.width;
    res.height = trackWindow_.height;

    return true;
}

bool TUMgoVision2D::updateImageSRV(tumgo_vision::srvImage::Request  &req, tumgo_vision::srvImage::Response &res)
{
    // Reduce the frame rate
    curr_msg_ = req.frame;
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
    InImage = filterImage(InImage,2);
    InImage = clahe(InImage,4,12);
    if(!process_image_)
        curr_image_ = InImage;
    if (show_raw_)
    {
        imshow("Camera frame",curr_image_);
        waitKey(1);
    }
    res.updated = true;
    return true;
}
/* *******************
 * Callback functions
 * *******************/
void TUMgoVision2D::updateImageCB(const sensor_msgs::ImageConstPtr &msg)
{
    curr_msg_ = *msg;
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
    if (show_images_)
    {
        imshow("RAW IMAGE",InImage);
        waitKey(1);
    }
    if(tracking_)
        trackWindow_ = tracking(trackWindow_);
}

/* *******************
 * Constructor
 * *******************/
TUMgoVision2D::TUMgoVision2D(ros::NodeHandle nh) : nh_(nh), priv_nh_("~")
{
    nh_.param<bool>("show_raw", show_raw_, true);
    nh_.param<bool>("show_images", show_images_, true);
    nh_.param<bool>("show_colors", show_colors_, false);
    nh_.param<bool>("synchronize", synchronize_, false);
    nh_.param<bool>("robot", robot_, true);
    cascades_pth_ = "src/team_Aplus/tumgo_vision/cascades";
    nh.param<std::string>("cascades", cascades_pth_, "src/team_Aplus/tumgo_vision/cascades");

    // Services
    detect_color_blob_service_ = nh_.advertiseService("/tumgo_vision/detect_color",&TUMgoVision2D::detectColorBlobSRV,this);
    detect_face_service_ = nh_.advertiseService("/tumgo_vision/detect_face",&TUMgoVision2D::detectFaceSRV,this);
    track_object_service_ = nh_.advertiseService("/tumgo_vision/track_object",&TUMgoVision2D::trackObjectSRV,this);
    update_image_service_ = nh_.advertiseService("/tumgo_vision/update_image",&TUMgoVision2D::updateImageSRV,this);
 
    get_image_client_ = nh_.serviceClient<tumgo_vision::srvGetImage>("/tumgo_vision/get_image");
    
    if (synchronize_)
        update_image_sub_ = nh_.subscribe("/tumgo_vision/update_image",1,&TUMgoVision2D::updateImageCB,this);
    else
    {
        // Subscribe to input video feed and publish output video feed.
        if (robot_)
           update_image_sub_ = nh_.subscribe("/xtion/rgb/image_raw",1,&TUMgoVision2D::updateImageCB,this);
        else
           update_image_sub_ = nh_.subscribe("/kinect2/qhd/image_color_rect",1,&TUMgoVision2D::updateImageCB,this);
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tumgo_vision2D");
    ros::NodeHandle n;
    ROS_INFO("Started TUMgo_vision2D node.");
    TUMgoVision2D node(n);
    ros::spin();
    return 0;
}


