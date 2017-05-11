/* ***************************************
    Author: Jianxiang Feng
    E-mail: jianxiang.feng@tum.de
    Author: Qiuhai Guo
    E-mail: qiuhai.guo@tum.de
    Author: Emilia Lozinska
    E-mail: e.lozinska@tum.de
*/

#include <tumgo_vision3D.h>

/* *******************
 * Local functions
 * *******************/
PointT TUMgoVision3D::convertPoint(PointT req)
{
    geometry_msgs::PointStamped point,point2;
    point.header.frame_id = "/xtion_rgb_optical_frame";
    point.point.x = req.x;
    point.point.y = req.y;
    point.point.z = req.z;
    ros::Time now = ros::Time::now();
    listener_->waitForTransform("/xtion_depth_optical_frame", "/base_footprint",
                              now, ros::Duration(5.0));
    //ROS_INFO("Transforming the points");
    listener_->transformPoint("/base_footprint", point, point2);
    // Return the result
    PointT res;
    res.x = point2.point.x;
    res.y = point2.point.y;
    res.z = point2.point.z;
    return res;
}

//! Callback for processing the Point Cloud data
void TUMgoVision3D::showCloud(PointCloudPtr plane, PointCloudPtr objects)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_two_clouds (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer_two_clouds->setBackgroundColor(0,0,0);

    viewer_two_clouds->removeAllPointClouds();
    viewer_two_clouds->removeAllShapes();
     // cloud: green / cloud2: red
    pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color1 (plane, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color2 (objects, 255, 0, 0);

    //add both
    viewer_two_clouds->addPointCloud<PointT> (plane, single_color1, "plane_cloud");
    viewer_two_clouds->addPointCloud<PointT> (objects, single_color2, "cluster_cloud");

    // set coordinateSystem and init camera
    viewer_two_clouds->addCoordinateSystem(1.0);
    viewer_two_clouds->initCameraParameters();

    while(!viewer_two_clouds->wasStopped())
    {
        viewer_two_clouds->spinOnce();
        boost::this_thread::sleep (boost::posix_time::microseconds(100000));
    }
    viewer_two_clouds->close();
}

void TUMgoVision3D::showCloud(PointCloudPtr plane, PointCloudPtr objects, std::vector<PointT> all_objects)
{
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_two_clouds (new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer_two_clouds->setBackgroundColor(0,0,0);

    viewer_two_clouds->removeAllPointClouds();
    viewer_two_clouds->removeAllShapes();
     // cloud: green / cloud2: red
    pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color1 (plane, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<PointT> single_color2 (objects, 255, 0, 0);

    //add both
    viewer_two_clouds->addPointCloud<PointT> (plane, single_color1, "plane_cloud");
    viewer_two_clouds->addPointCloud<PointT> (objects, single_color2, "cluster_cloud");

    // set coordinateSystem and init camera
    viewer_two_clouds->addCoordinateSystem(1.0);
    viewer_two_clouds->initCameraParameters();
    std::ostringstream name;
    for (int i = 0; i < all_objects.size(); i++)
    {
        name << "cube" << i;
        viewer_two_clouds->addCube(all_objects[i*2].x, all_objects[i*2+1].x, all_objects[i*2].y, all_objects[i*2+1].y, all_objects[i*2].z, all_objects[i*2+1].z, 1.0,1.0,1.0,name.str());
    }

    while(!viewer_two_clouds->wasStopped())
    {
        viewer_two_clouds->spinOnce();
        boost::this_thread::sleep (boost::posix_time::microseconds(100000));
    }
    viewer_two_clouds->close();
}

PointCloudPtr TUMgoVision3D::segmentTable(pcl::PointCloud<PointT>::Ptr input, float per)
{
    pcl::PointCloud<PointT>::Ptr cloud_f( new pcl::PointCloud<PointT> ); // cloud to store the filter the data
    pcl::copyPointCloud<PointT,PointT>(*input, *cloud_f);

    pcl::PointCloud<PointT>::Ptr cloud_p( new pcl::PointCloud<PointT> ); // cloud to store the main plane cloud
    pcl::PointCloud<PointT>::Ptr cloud_plane( new pcl::PointCloud<PointT> ); // cloud to store the main plane cloud
    pcl::PointCloud<PointT>::Ptr cloud_np( new pcl::PointCloud<PointT> ); // cloud to store the non-main plane cloud

    // Create the segmentation object for the plane model and set all the parameters using pcl::SACSegmentation<PointT>
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers( new pcl::PointIndices );
    pcl::ModelCoefficients::Ptr coefficients( new pcl::ModelCoefficients );

    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (3000);
    seg.setDistanceThreshold (0.01);

    // Create the filtering object
    pcl::ExtractIndices<PointT> extract;

    int nr_points = cloud_f->points.size();
    while (cloud_f->points.size () > per * nr_points)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_f);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            return PointCloudPtr();
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud (cloud_f);
        extract.setIndices (inliers);
        extract.setNegative (false);
        // Get the points associated with the planar surface
        extract.filter (*cloud_plane);
        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_np);
        *cloud_f = *cloud_np;
        *cloud_p += *cloud_plane; 
    }
    showCloud(cloud_p,cloud_f);
    curr_tables_ = cloud_p;
    return cloud_f;
}

string TUMgoVision3D::shapeDetector(pcl::PointCloud<PointT>::Ptr input)
{
    // Store maximum size of inliers and shape name
    int max_inliers = 0;
    string shape;

    pcl::PointCloud<PointT>::Ptr cloud_f( new pcl::PointCloud<PointT> ); // cloud to store the filter the data
    pcl::copyPointCloud<PointT,PointT>(*input, *cloud_f);

    pcl::PointCloud<PointT>::Ptr cloud_p( new pcl::PointCloud<PointT> ); // cloud to store the main plane cloud
    pcl::PointCloud<PointT>::Ptr cloud_plane( new pcl::PointCloud<PointT> ); // cloud to store the main plane cloud
    pcl::PointCloud<PointT>::Ptr cloud_np( new pcl::PointCloud<PointT> ); // cloud to store the non-main plane cloud

    // BOX
    ROS_INFO("Box");
    // Create the segmentation object for the plane model and set all the parameters using pcl::SACSegmentation<PointT>
    pcl::SACSegmentation<PointT> seg_box;
    pcl::PointIndices::Ptr inliers( new pcl::PointIndices );
    pcl::ModelCoefficients::Ptr coefficients( new pcl::ModelCoefficients );
    // Optional
    seg_box.setOptimizeCoefficients (true);
    // Mandatory
    seg_box.setModelType (pcl::SACMODEL_PLANE);
    seg_box.setMethodType (pcl::SAC_RANSAC);
    seg_box.setMaxIterations (3000);
    seg_box.setDistanceThreshold (0.01);
    seg_box.setInputCloud (cloud_f);
    seg_box.segment (*inliers, *coefficients);
    if (inliers->indices.size () > max_inliers)
    {
        max_inliers = inliers->indices.size ();
        shape = "Box";
    }

    // // CYLINDER
    // ROS_INFO("Cyllinder check");
    // // Create the segmentation object for the plane model and set all the parameters using pcl::SACSegmentation<PointT>
    // pcl::SACSegmentation<PointT> seg_cylinder;
    // // Optional
    // seg_cylinder.setOptimizeCoefficients (true);
    // // Mandatory
    // seg_cylinder.setModelType (pcl::SACMODEL_CYLINDER); 
    // seg_cylinder.setMethodType (pcl::SAC_RANSAC);
    // seg_cylinder.setMaxIterations (3000);
    // seg_cylinder.setDistanceThreshold (0.01);
    // seg_cylinder.setNormalDistanceWeight (0.1);
    // seg_cylinder.setRadiusLimits (0, 0.1);
    // seg_cylinder.setInputCloud (cloud_f);
    // seg_cylinder.segment (*inliers, *coefficients);s
    // if (inliers->indices.size () > max_inliers)
    // {
    //     max_inliers = inliers->indices.size ();
    //     shape = "Cylinder";
    // }


    // SPHERE
    ROS_INFO("Sphere");
    // Create the segmentation object for the plane model and set all the parameters using pcl::SACSegmentation<PointT>
    pcl::SACSegmentation<PointT> seg_sphere;
    // Optional
    seg_sphere.setOptimizeCoefficients (true);
    // Mandatory
    seg_sphere.setModelType (pcl::SACMODEL_SPHERE);
    seg_sphere.setMethodType (pcl::SAC_RANSAC);
    seg_sphere.setMaxIterations (3000);
    seg_sphere.setDistanceThreshold (0.01);
    seg_sphere.setInputCloud (cloud_f);
    seg_sphere.segment (*inliers, *coefficients);
    if (inliers->indices.size () > max_inliers)
    {
        max_inliers = inliers->indices.size ();
        shape = "Sphere";
    }

    return shape;
}

bool TUMgoVision3D::segmentCloud(PointCloudPtrVec &cloud_cluster)
{
    ///local variants for saving point cloud
    vector<PointCloudPtr, Eigen::aligned_allocator< PointCloud > > cloud_plane;//for saving all found planes cluster
    PointCloudPtr cloud_filtered_dsample(new PointCloud );// for saving the filtered point cloud
    PointCloudPtr cloud_filtered_NaN(new PointCloud );// for saving the filtered NaN point cloud
    PointCloudPtr cloud_non_plane_cluster(new PointCloud);//for saving the non-plane cluster

    ///pcl methods declaration
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::ExtractIndices<PointT> extract;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);//normals for searching for plane
    pcl::PassThrough< PointT > pass;


    PointCloudPtr cloud = curr_cloud_.makeShared(); // put cloud data into heap and get the poiter
    std::cout << "PointCloud before filtering has: " << curr_cloud_.points.size() << " data points." << std::endl;
    std::cout << "width: " << curr_cloud_.width << " height: " << curr_cloud_.height << std::endl;


    /// Build a passthrough filter to remove spurious NaNs
    pass.setInputCloud (cloud);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0, 3.0);
    pass.filter (*cloud_filtered_NaN);
    std::cout << "PointCloud after filtering NaN has: " << cloud_filtered_NaN->points.size () << " data points." << std::endl;


    /// Down sample the pointcloud using VoxelGrid
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud (cloud_filtered_NaN);
    sor.setLeafSize (down_sampling_voxel_size, down_sampling_voxel_size, down_sampling_voxel_size);
    //sor.setDownsampleAllData(true); //trick for RGB
    sor.filter (*cloud_filtered_dsample);
    std::cout << "PointCloud after filtering has: " << cloud_filtered_dsample->points.size()  << " data points." << std::endl;


    /// Create the segmentation object for the plane model and cylinder model
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
    pcl::PointIndices::Ptr inliers( new pcl::PointIndices );
    pcl::ModelCoefficients::Ptr coefficients( new pcl::ModelCoefficients );

    /// set parameters of the SACS plane segmentation
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight (0.1);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (100);
    seg.setDistanceThreshold (0.02);

    ///start segmenting plane
    int i = 0, nr_points = (int) cloud_filtered_dsample->points.size ();
    // Estimate point normals of the downsampled point cloud
    ne.setSearchMethod (tree);
    ne.setInputCloud (cloud_filtered_dsample);
    ne.setKSearch (50);
    ne.compute (*cloud_normals);
    std::cout << "cloud_normals after filtering has: " << cloud_normals->points.size () << " data points." << std::endl;
    //extract the plane
    while (cloud_filtered_dsample->points.size () > 0.15 * nr_points)
    {
        seg.setInputCloud (cloud_filtered_dsample);
        seg.setInputNormals (cloud_normals);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            std::cout << "Could not estimate a planar model after "<<i<<" times extractions" << std::endl;
            break;
        }


        // Extract the inliers from point cloud and store it into vector cloud_plane
        PointCloudPtr cloud_temp(new PointCloud);
        extract.setInputCloud (cloud_filtered_dsample);
        extract.setIndices (inliers);
        extract.setNegative (false);//false means extract the inliers
        extract.filter (*cloud_temp);
        cloud_plane.push_back(cloud_temp);

        // Remove the planar inliers, extract the rest
        extract.setNegative (true);//true means extract the data except for inliers
        extract.filter (*cloud_non_plane_cluster);
        cloud_filtered_dsample.swap(cloud_non_plane_cluster);


        //update normals
        extract_normals.setNegative (true);
        extract_normals.setInputCloud (cloud_normals);
        extract_normals.setIndices (inliers);
        extract_normals.filter(*cloud_normals);
        i++;
    }
    if(!cloud_plane.empty()){
        plane_pub_.publish(*cloud_plane[0]); //publish the the first found plane which is the largest one
        //pass.setInputCloud (cloud_filtered_dsample);
        //pass.setFilterFieldName ("x");
        //pass.setFilterLimits (0, 4.0);
        //pass.filter (*cloud_filtered_dsample);
        clusters_pub_.publish(*cloud_filtered_dsample);
        //!assign it to the class parameter
        curr_table_ = cloud_plane[0];
    }

    std::cout << "PointCloud for extracting cylinder has: " << cloud_filtered_dsample->points.size()  << " data points." << std::endl;


    ///use Euclidean distance as metric to separate the cluster
    tree->setInputCloud(cloud_filtered_dsample);
    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (0.02); // 2cm
    ec.setMinClusterSize (80);
    ec.setMaxClusterSize (25000);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud_filtered_dsample);
    ec.extract (cluster_indices);
    std::cout << "cluster_indices has: " << cluster_indices.size() << std::endl;
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it){
        PointCloudPtr temp(new PointCloud);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit){
            temp->points.push_back(cloud_filtered_dsample->points[*pit]);
        }
        temp->width=temp->size(); temp->height= 1;
        cloud_cluster.push_back (temp);
    }
    for(unsigned i = 0; i<cloud_cluster.size();i++){
        std::cout << "cloud_cluster["<<i<<"] has: " << cloud_cluster[i]->size() << std::endl;
    }
    if(!cloud_cluster.empty()){
        sensor_msgs::PointCloud2::Ptr pc2_msg(new sensor_msgs::PointCloud2);
        pcl::toROSMsg( *cloud_cluster[1], *pc2_msg);
        pc2_msg->header.frame_id = "/xtion_depth_optical_frame";
        clusters_ec_pub_.publish(*pc2_msg);
        //pcl::io::savePCDFileASCII("apple.pcd", *cloud_cluster[1]);
        return true;
    }
    else return false;
}

bool TUMgoVision3D::detectObject(PointCloudPtrVec &cloud_cluster)
{
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    ///load the model of required object
    if(!load_model){
        std::string label = "cocacola_wide";
        if (pcl::io::loadPCDFile (models_dir+label+".pcd", model_list_["Cylinder"]) < 0)
        {
            std::cout << "Error loading "<<label<< ".pcd" <<"\n"<< std::endl;
            return false;
        }
        label = "ball";
        if (pcl::io::loadPCDFile (models_dir+label+".pcd", model_list_["Sphere"]) < 0)
        {
            std::cout << "Error loading "<<label<< ".pcd" <<"\n"<< std::endl;
            return false;
        }
        label = "apple";
        if (pcl::io::loadPCDFile (models_dir+label+".pcd", model_list_["Apple"]) < 0)
        {
            std::cout << "Error loading "<<label<< ".pcd" <<"\n"<< std::endl;
            return false;
        }
        label = "orange";
        if (pcl::io::loadPCDFile (models_dir+label+".pcd", model_list_["Orange"]) < 0)
        {
            std::cout << "Error loading "<<label<< ".pcd" <<"\n"<< std::endl;
            return false;
        }
        label = "tall_cube";
        if (pcl::io::loadPCDFile (models_dir+label+".pcd", model_list_["Box"]) < 0)
        {
            std::cout << "Error loading "<<label<< ".pcd" <<"\n"<< std::endl;
            return false;
        }
        std::cout<<"\n"<<"load all models" <<" successfully"<<std::endl;
        load_model = true;
    }
    std::cout<<"cloud_cluster size "<<cloud_cluster.size()<<std::endl;
    
    
    ///compute the normal of extracted cluster vector
    pcl::NormalEstimationOMP<PointT, NormalT> norm_est;
    std::vector<pcl::PointCloud<NormalT>::Ptr , Eigen::aligned_allocator<pcl::PointCloud<NormalT> > > objects_normals;
    for(unsigned i = 0; i<cloud_cluster.size(); i++){
        pcl::PointCloud<NormalT>::Ptr temp_objects_normals (new pcl::PointCloud<NormalT>);
        norm_est.setKSearch (10);
        norm_est.setInputCloud (cloud_cluster[i]);
        norm_est.compute (*temp_objects_normals);
        objects_normals.push_back(temp_objects_normals);
        //std::cout<<"objects_normals "<<i<<" size "<<temp_objects_normals->size()<<std::endl;
    }
    ///extract DescriptorT Feature from extracted cluster vector
    pcl::FPFHEstimation<PointT, NormalT, DescriptorT> fpfh;
    fpfh.setSearchMethod (tree);
    fpfh.setRadiusSearch (0.05);
    std::vector<pcl::PointCloud<DescriptorT>::Ptr , Eigen::aligned_allocator<pcl::PointCloud<DescriptorT> > > objects_descriptors;
    for(unsigned i = 0; i<cloud_cluster.size(); i++){
        pcl::PointCloud<DescriptorT>::Ptr temp_objects_descriptors (new pcl::PointCloud<DescriptorT>);
        fpfh.setInputCloud (cloud_cluster[i]);
        fpfh.setInputNormals (objects_normals[i]);
        fpfh.compute (*temp_objects_descriptors);
        objects_descriptors.push_back(temp_objects_descriptors);
        std::cout<<"objects "<<i<<" feature point size "<<objects_descriptors[i]->size()<<std::endl;
    }
    std::cout<<"\n"<<std::endl;
    //flag for marking first assignment to object_list_
    int flag = 1;
    object_list_.clear();
    for(std::map<std::string, PointCloud>::iterator it_model_list = model_list_.begin(); it_model_list!=model_list_.end(); ++it_model_list)
    {
        std::cout<<"comparing with model "<<it_model_list->first<<std::endl;
        PointCloudPtr model = it_model_list->second.makeShared();
        ///compute normal of model
        pcl::PointCloud<NormalT>::Ptr model_normals (new pcl::PointCloud<NormalT>);
        norm_est.setKSearch (10);
        norm_est.setInputCloud (model);
        norm_est.compute (*model_normals);

        ///extract DescriptorT Feature from model
        pcl::PointCloud<DescriptorT>::Ptr model_descriptors (new pcl::PointCloud<DescriptorT>);
        fpfh.setInputCloud (model);
        fpfh.setInputNormals (model_normals);
        fpfh.compute (*model_descriptors);
        std::cout<<"model feature point size "<<model_descriptors->size()<<std::endl;

        ///matching
        std::vector<pcl::CorrespondencesPtr , Eigen::aligned_allocator<pcl::Correspondences > > model_object_corrs;
        pcl::KdTreeFLANN<DescriptorT> match_search;
        ///find the object corresponding to the model from the extracted cluster vector
        match_search.setInputCloud (model_descriptors);
        for(unsigned i = 0; i<cloud_cluster.size(); i++)
        {
            pcl::CorrespondencesPtr temp_model_object_corrs (new pcl::Correspondences);
            for (size_t j = 0; j < objects_descriptors[i]->size(); ++j)
            {
              std::vector<int> neigh_indices (1);
              std::vector<float> neigh_sqr_dists (1);
              int found_neighs = match_search.nearestKSearch (objects_descriptors[i]->at (j), 1, neigh_indices, neigh_sqr_dists);
              //std::cout<<neigh_sqr_dists[0]<<std::endl;
              if(found_neighs == 1 && neigh_sqr_dists[0] < 350.0f) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
              {
                pcl::Correspondence corr (neigh_indices[0], static_cast<int> (j), neigh_sqr_dists[0]);
                temp_model_object_corrs->push_back (corr);
              }
            }
            model_object_corrs.push_back(temp_model_object_corrs);
            std::cout << "model and "<<i<<"th cluster have found "<< model_object_corrs[i]->size () <<" Correspondences.\n"<< std::endl;
            if(flag){
                ObjectType object_temp;
                object_temp.label = it_model_list->first;
                object_temp.confidence = float(model_object_corrs[i]->size())/float(model->size());
                //std::cout << "object_temp.confidence"<<object_temp.confidence<<std::endl;
                object_temp.obj_pc = *cloud_cluster[i];
                object_list_.push_back(object_temp);
            }
            else{
                float confidence_temp;
                confidence_temp = float(model_object_corrs[i]->size())/float(model->size());
                if(object_list_[i].confidence < confidence_temp){
                    object_list_[i].confidence = float(model_object_corrs[i]->size())/float(model->size());
                    object_list_[i].label = it_model_list->first;
                }
            }
        }
        flag = 0;
    
    }
    int temp_flag = 0;
    for(unsigned i = 0; i<object_list_.size(); ++i){
        if(object_list_[i].confidence>0) temp_flag++;
    }
    if(temp_flag){
        return true;
    }
    else return false;
}

void TUMgoVision3D::processObject(PointCloud object_pc, 
                                        float &width, 
                                        float &depth, 
                                        float &height, 
                                        float &x, 
                                        float &y, 
                                        float &z,  
                                        PointT &min, 
                                        PointT &max)
{
    Eigen::Vector4f centroid;
    // calculate the bounding box
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(object_pc, min_pt, max_pt);
    width = max_pt[0] - min_pt[0];     
    height = max_pt[1] - min_pt[1];     
    depth = max_pt[2] - min_pt[2];
    // Get points for color extraction
    PointT min_p, max_p;
    pcl::getMinMax3D(object_pc, min_p, max_p);
    min = min_p;
    max = max_p;
    // calculate the center
    //x = (max_pt[0] + min_pt[0]) / 2.0;
    //y = (max_pt[1] + min_pt[1]) / 2.0;
    //z = (max_pt[2] + min_pt[2]) / 2.0;
    // use centroid as center
    pcl::compute3DCentroid(object_pc , centroid );
    float d = sqrt(centroid(0)*centroid(0) + centroid(1)*centroid(1) + centroid(1)*centroid(1));
    PointT point;
    point.x = centroid(0);
    point.y = centroid(1);
    point.z = centroid(2);
    //cout << "x,y,z before converting to base_footprint: " << point.x << ", " << point.y  << ", " << point.z << std::endl;
    point = convertPoint(point);
    //cout << "x,y,z after converting to base_footprint: " << point.x << ", " << point.y  << ", " << point.z << std::endl;
    x = point.x;
    y = point.y;
    z = point.z;
    process_cloud_ = false;
    return;
}

/* *******************
 * Service functions
 * *******************/
bool TUMgoVision3D::detectObjectSRV(tumgo_vision::srvDetectObject3D::Request  &req, tumgo_vision::srvDetectObject3D::Response &res)
{
    //!segment the point cloud into plane and cylinder and save them into the class private parameters: curr_table_ and curr_object_
    //if segmentation is done, rec_obj would be true, otherwise false
    PointCloudPtrVec cloud_cluster;
    rec_obj = false;
    if(segmentCloud(cloud_cluster)){
        rec_obj = detectObject(cloud_cluster);
    }
    if(rec_obj){
        ///processing the found object
        std::cout<<"object_list_ size: "<<object_list_.size()<<std::endl;
        curr_object_poses_.poses.clear();
        std::vector<string> label_temp;
        for(unsigned i = 0; i<object_list_.size(); ++i)
        {
            std::cout<<"object_list_["<<i<<"].confidence: "<<object_list_[i].confidence<<std::endl;
            if(object_list_[i].confidence>0)
            {
                //label_temp.push_back(object_list_[0].label);
                if(std::find(label_temp.begin(), label_temp.end(), object_list_[i].label) == label_temp.end()){
                    label_temp.push_back(object_list_[i].label);
                    for(unsigned j = i; j<object_list_.size(); j++){
                        if(object_list_[i].label == object_list_[j].label){
                                std::cout<<"object "<<object_list_[i].label<<" is found with confidence: "<<object_list_[i].confidence<<std::endl;
                                processObject(object_list_[i].obj_pc, 
                                                    object_list_[i].width,
                                                    object_list_[i].depth, 
                                                    object_list_[i].height, 
                                                    object_list_[i].x, 
                                                    object_list_[i].y, 
                                                    object_list_[i].z,
                                                    object_list_[i].min,
                                                    object_list_[i].max);
                                res.label.push_back(object_list_[i].label);
                                res.confidence.push_back(object_list_[i].confidence);
                                res.width.push_back(object_list_[i].width);
                                res.depth.push_back(object_list_[i].depth);
                                res.height.push_back(object_list_[i].height);
                                res.x.push_back(object_list_[i].x);
                                res.y.push_back(object_list_[i].y);
                                res.z.push_back(object_list_[i].z);
                                res.min_x.push_back(object_list_[i].min.x);
                                res.min_y.push_back(object_list_[i].min.y);
                                res.min_z.push_back(object_list_[i].min.z);
                                res.max_x.push_back(object_list_[i].max.x);
                                res.max_y.push_back(object_list_[i].max.y);
                                res.max_z.push_back(object_list_[i].max.z);
                                geometry_msgs::Pose pose_temp;
                                pose_temp.position.x = object_list_[i].x;
                                pose_temp.position.y = object_list_[i].y;
                                pose_temp.position.z = object_list_[i].z;
                                pose_temp.orientation.x = 0;
                                pose_temp.orientation.y = 0;
                                pose_temp.orientation.z = 0;
                                pose_temp.orientation.w = 1;
                                curr_object_poses_.poses.push_back(pose_temp);
                        }
                    }
                }
            }
        }
        res.result = true;

        curr_object_poses_.header.frame_id = "/base_footprint";
        curr_object_poses_.header.stamp = ros::Time::now();
        object_pose_pub_.publish( curr_object_poses_);
    }
    else {
        res.result = false;
        ROS_INFO_STREAM("Object not found");
    }
    return true;
}

bool TUMgoVision3D::getFacePositionSRV(tumgo_vision::srvGetFacePosition::Request &req, tumgo_vision::srvGetFacePosition::Response &res)
{
    tumgo_vision::srvDetectFace srv;
    // srv.request.personID = req.personID;
    Eigen::Vector4f centroid;
    PointCloud temp_head;
    if(detect_face_client_.call(srv)){
        if(srv.response.result){
            std::cout << "required face detected successfully!"<<std::endl;
            res.result = true;
            //extract the head pointcloud
            for(int i = srv.response.y-25; i < (srv.response.y+srv.response.height+2); i++){
                for(int j = srv.response.x; j < (srv.response.x+srv.response.width+2); j++){
                    temp_head.points.push_back(curr_cloud_.at(i*640+j));
                }
            }
            pcl::compute3DCentroid(temp_head , centroid );
            float d = sqrt(centroid(0)*centroid(0) + centroid(1)*centroid(1) + centroid(1)*centroid(1));
            pcl::PassThrough<PointT> pass;
            pass.setInputCloud (temp_head.makeShared());
            pass.setFilterFieldName ("z");
            pass.setFilterLimits (1, d+1);
            pass.filter (temp_head);
            PointT point;
            point.x = centroid(0);
            point.y = centroid(1);
            point.z = centroid(2);
            point = convertPoint(point);
            res.x = point.x;
            res.y = point.y;
            res.z = point.z;
            res.personID = srv.response.personID;
            sensor_msgs::PointCloud2 pc2_msg;
            pcl::toROSMsg(temp_head, pc2_msg);
            pc2_msg.header.frame_id = "/xtion_depth_optical_frame";
            res.cloud = pc2_msg;
        }
        else{
            std::cout << "required face is not detected!"<<std::endl;
            res.result = false;
        }
    }
    else std::cout << "detectFaceSRV can not be called! "<<std::endl;
    return true;
}

bool TUMgoVision3D::segmentPlaneSRV(tumgo_vision::srvSegmentPlanes::Request  &req, tumgo_vision::srvSegmentPlanes::Response &res)
{
    if (curr_cloud_.empty())
    {
        ROS_ERROR("No cloud to segment!");
        return false;
    }
    PointCloudPtrVec planes; //for saving planes used for control 
    pcl::PointCloud<PointT>::Ptr cloud = curr_cloud_.makeShared(); // cloud to operate
    pcl::PointCloud<PointT>::Ptr cloud_f( new pcl::PointCloud<PointT> ); // cloud to store the filter the data

    pcl::PointCloud<PointT>::Ptr cloud_p( new pcl::PointCloud<PointT> ); // cloud to store the main plane cloud
    pcl::PointCloud<PointT>::Ptr cloud_plane( new pcl::PointCloud<PointT> ); // cloud to store the main plane cloud

    pcl::PointCloud<PointT>::Ptr cloud_np( new pcl::PointCloud<PointT> ); // cloud to store the non-main plane cloud

    // Down sample the pointcloud using VoxelGrid
    pcl::VoxelGrid<PointT> vog;
    vog.setInputCloud (cloud);
    vog.setLeafSize (0.01f, 0.01f, 0.01f); // 1 cm
    vog.filter (*cloud_f);

    // Create the segmentation object for the plane model and set all the parameters using pcl::SACSegmentation<PointT>
    pcl::SACSegmentation<PointT> seg;
    pcl::PointIndices::Ptr inliers( new pcl::PointIndices );
    pcl::ModelCoefficients::Ptr coefficients( new pcl::ModelCoefficients );

    // Optional
    seg.setOptimizeCoefficients (true);
    // Mandatory
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (2000);
    seg.setDistanceThreshold (0.015);

    // Create the filtering object
    pcl::ExtractIndices<PointT> extract;

    int nr_points = cloud_f->points.size();
    while (cloud_f->points.size () > percent * nr_points)
    {
        // Segment the largest planar component from the remaining cloud
        seg.setInputCloud (cloud_f);
        seg.segment (*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
            return false;
        }

        // Extract the planar inliers from the input cloud
        pcl::ExtractIndices<PointT> extract;
        extract.setInputCloud (cloud_f);
        extract.setIndices (inliers);
        extract.setNegative (false);
        // Get the points associated with the planar surface
        extract.filter (*cloud_plane);
        // Remove the planar inliers, extract the rest
        extract.setNegative (true);
        extract.filter (*cloud_np);
        *cloud_f = *cloud_np;
        planes.push_back(cloud_plane);
        *cloud_p += *cloud_plane; 
    }

    // if(!planes.empty()){
    //     for(int i = 0; i<planes.size(); i++){
    //          ObjectType temp_planes;
    //          processObject(*planes[i], 
    //                      temp_planes.width, 
    //                      temp_planes.depth, 
    //                      temp_planes.height, 
    //                      temp_planes.x, 
    //                      temp_planes.y, 
    //                      temp_planes.z);
    //          res.planes.push_back("plane"+char(i));
    //          res.x.push_back(temp_planes.x);
    //          res.y.push_back(temp_planes.y);
    //          res.z.push_back(temp_planes.z);
    //          res.width.push_back(temp_planes.width);
    //          res.height.push_back(temp_planes.height);
    //          res.depth.push_back(temp_planes.depth);
    //      }
    //     res.result = true;
    // }
    // else res.result = false;

    curr_plane_ = cloud_p;
    curr_objects_ = cloud_np;
    if (show_cloud_)
        showCloud(curr_plane_,curr_objects_);
    curr_objects_ = segmentTable(curr_objects_,0.5);

    PointT table_min, table_max, plane_min, plane_max;

    pcl::getMinMax3D (*curr_plane_, table_min, table_max);
    pcl::getMinMax3D (*curr_tables_, plane_min, plane_max);

    res.planes.push_back("table");
    res.x.push_back((table_max.x-table_min.x)/2);
    res.result = true;
    res.y.push_back((table_max.y-table_min.y)/2);
    res.z.push_back((table_max.z-table_min.z)/2); 
    res.width.push_back(abs(table_max.x-table_min.x));
    res.height.push_back(abs(table_max.y-table_min.y));
    res.depth.push_back(abs(table_max.z-table_min.z)); 

    res.planes.push_back("wall");
    res.x.push_back((plane_max.x-plane_min.x)/2);
    res.y.push_back((plane_max.y-plane_min.y)/2);
    res.z.push_back((plane_max.z-plane_min.z)/2); 
    res.width.push_back(abs(plane_max.x-plane_min.x));
    res.height.push_back(abs(plane_max.y-plane_min.y));
    res.depth.push_back(abs(plane_max.z-plane_min.z)); 
    return true;
}

bool TUMgoVision3D::segmentObjectsSRV(tumgo_vision::srvSegmentObjects::Request  &req, tumgo_vision::srvSegmentObjects::Response &res)
{
    if (curr_cloud_.empty())
    {
        ROS_ERROR("No cloud to segment!");
        return false;
    }   
    if ((*curr_plane_).empty())
    {
        ROS_ERROR("Segment the planes first!");
        return false;
    }
    all_objects_.clear();
    all_points_.clear();

    std::vector<float> vector_min_x;
    std::vector<float> vector_min_y;
    std::vector<float> vector_min_z;
    std::vector<float> vector_max_x;
    std::vector<float> vector_max_y;
    std::vector<float> vector_max_z;
    std::vector<float> center_x;
    std::vector<float> center_y;
    std::vector<float> center_z;
    std::vector<float> height;
    std::vector<float> width;
    std::vector<float> depth;
    std::vector<int> vector_ind_x;
    std::vector<int> vector_ind_y;

    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT>);
    tree->setInputCloud(curr_objects_);

    std::vector<pcl::PointIndices> cluster_indices;

    pcl::EuclideanClusterExtraction<PointT> ec;
    ec.setClusterTolerance (tolerance);
    ec.setMinClusterSize (min_size);
    ec.setMaxClusterSize (max_size);

    ec.setInputCloud (curr_objects_);
    ec.extract (cluster_indices);

    std::vector<PointT> new_points;
    
    ROS_INFO("Found %lu objects", cluster_indices.size());

    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {

        // Cloud cluster for the current object
        pcl::PointCloud<PointT>::Ptr cloud_cluster (new pcl::PointCloud<PointT>);
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
            cloud_cluster->points.push_back (curr_objects_->points[*pit]); //*
        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // Bounding boxes
        pcl::PCA< PointT > pca;
        pcl::PointCloud< PointT > proj;
        pca.setInputCloud (cloud_cluster);
        pca.project (*cloud_cluster, proj);

        PointT proj_min;
        PointT proj_max;
        Eigen::Vector4f centroid;
        pcl::getMinMax3D (*cloud_cluster, proj_min, proj_max);

        // PointT min;
        // PointT max;
        // pca.reconstruct (proj_min, min);
        // pca.reconstruct (proj_max, max);

        if ( abs(proj_max.x - proj_min.x) > 0.03 && abs(proj_max.y - proj_min.y) > 0.03)
        {
            if ( abs(proj_max.x - proj_min.x) < 0.25 * cloud_cluster->width && abs(proj_max.y - proj_min.y) < 0.25 * cloud_cluster->height)
            {
                // If the object is of appropriate size, store it
                all_objects_.push_back(cloud_cluster);
                all_points_.push_back(proj_min);
                all_points_.push_back(proj_max);
                // Save for the response
                vector_min_x.push_back(proj_min.x);
                vector_min_y.push_back(proj_min.y);
                vector_min_z.push_back(proj_min.z);
                vector_max_x.push_back(proj_max.x);
                vector_max_y.push_back(proj_max.y);
                vector_max_z.push_back(proj_max.z); 
                center_x.push_back((proj_max.x-proj_min.x)/2);
                center_y.push_back((proj_max.y-proj_min.y)/2);
                center_z.push_back((proj_max.z-proj_min.z)/2); 
                width.push_back(abs(proj_max.x-proj_min.x));
                height.push_back(abs(proj_max.y-proj_min.y));
                depth.push_back(abs(proj_max.z-proj_min.z)); 
            } 
            else
                ROS_INFO ("Too big object found");
        }
        else
            ROS_INFO ("Too small object found");
    }

    res.min_x = vector_min_x;
    res.min_y = vector_min_y;
    res.min_z = vector_min_z;
    res.max_x = vector_max_x;
    res.max_y = vector_max_y;
    res.max_z = vector_max_z;
    res.center_x = center_x;
    res.center_y = center_y;
    res.center_z = center_z;
    res.width = width;
    res.height = height;
    res.depth = depth;
    if (show_cloud_)
        showCloud(curr_plane_,curr_objects_,all_points_);
    return true;
}

bool TUMgoVision3D::recognizeObjectsSRV(tumgo_vision::srvRecognition3D::Request  &req, tumgo_vision::srvRecognition3D::Response &res)
{
    if (curr_cloud_.empty())
    {
        ROS_ERROR("No cloud to segment!");
        return false;
    }   
    if ((*curr_objects_).empty())
    {
        ROS_ERROR("Segment the planes first!");
        return false;
    }
    if (all_objects_.empty())
    {
        ROS_ERROR("Segment the objects first!");
        return false;
    }
    for (int i = 0; i < all_objects_.size(); i++)
    {
        res.label.push_back(shapeDetector(all_objects_[i]));        
        PointT proj_min;
        PointT proj_max;
        pcl::getMinMax3D (*all_objects_[i], proj_min, proj_max);
        res.x.push_back( (proj_max.x - proj_min.x) / 2);
        res.y.push_back( (proj_max.y - proj_min.y) / 2);
        res.z.push_back( (proj_max.z - proj_min.z) / 2);
        res.width.push_back( abs(proj_max.x - proj_min.x) );
        res.height.push_back( abs(proj_max.y - proj_min.y) );
        res.depth.push_back( abs(proj_max.z - proj_min.z) );
    }
    return true;
}

bool TUMgoVision3D::updateCloudSRV(tumgo_vision::srvCloud::Request  &req, tumgo_vision::srvCloud::Response &res)
{
    if(!process_cloud_)
    {
        pcl::fromROSMsg(req.cloud,curr_cloud_);
        curr_pcl_ = req.cloud;
        res.updated = true;
    }
    else
        res.updated = false;
    return true;
}

/* *******************
 * Callback functions
 * *******************/

void TUMgoVision3D::updateCloudCB(const sensor_msgs::PointCloud2::ConstPtr &msg)
{

    if(!process_cloud_)
    {
        //ROS_INFO("Saving the cloud.");
        pcl::fromROSMsg(*msg,curr_cloud_);
        curr_pcl_ = *msg;
        //ROS_INFO("Saved the cloud.");
    }
}

// Constructor
TUMgoVision3D::TUMgoVision3D(ros::NodeHandle nh, std::string processing_frame, tf::TransformListener *listener) : 
                            nh_(nh), priv_nh_("~"),
                            listener_(listener),
                            processing_frame_(processing_frame),
                            process_cloud_(false)
{
    nh_.param<bool>("show_cloud", show_cloud_, true);
    nh_.param<bool>("synchronize", synchronize_, false);
    nh_.param<bool>("robot", robot_, true);
    nh_.param<std::string>("models_dir", models_dir, "/home/rcah/ros/workspaces/project_ws/src/tumgo_vision3D/src/models/");


    // Services
    detect_object_service_ = nh_.advertiseService("/tumgo_vision/detect_object3D",&TUMgoVision3D::detectObjectSRV,this);
    update_cloud_service_ = nh_.advertiseService("/tumgo_vision/update_cloud",&TUMgoVision3D::updateCloudSRV,this);
    segment_plane_service_ = nh_.advertiseService("/tumgo_vision/segment_plane",&TUMgoVision3D::segmentPlaneSRV,this);
    segment_objects_service_ = nh_.advertiseService("/tumgo_vision/segment_object",&TUMgoVision3D::segmentObjectsSRV,this);
    recognition_service_ = nh_.advertiseService("/tumgo_vision/recognition3D",&TUMgoVision3D::recognizeObjectsSRV,this);
    get_face_position_service_ = nh_.advertiseService("/tumgo_vision/get_face_position",&TUMgoVision3D::getFacePositionSRV,this);
    
    //clients
    detect_face_client_ = nh_.serviceClient<tumgo_vision::srvDetectFace>("/tumgo_vision/detect_face");

    // Subscribers
    //update_cloud_sub_ = nh_.subscribe("/xtion/depth_registered/points",1,&TUMgoVision3D::updateCloudCB,this);
    if (synchronize_)
        update_cloud_sub_ = nh_.subscribe("/tumgo_vision/update_cloud",1,&TUMgoVision3D::updateCloudCB,this);
    else
    {
        // Subscribe to input video feed and publish output video feed.
        if (robot_)
           update_cloud_sub_ = nh_.subscribe("/xtion/depth_registered/points",1,&TUMgoVision3D::updateCloudCB,this);
        else
           update_cloud_sub_ = nh_.subscribe("/kinect2/qhd/points",1,&TUMgoVision3D::updateCloudCB,this);
    }

    // Publishers
    plane_pub_ = nh_.advertise< PointCloud >("/tumgo_vision/plane_points", 10);
    clusters_pub_ = nh_.advertise< PointCloud >("/tumgo_vision/non_plane_points", 10);
    object_pose_pub_ = nh_.advertise< geometry_msgs::PoseArray >("tumgo_vision/pose", 10);
    clusters_ec_pub_ = nh_.advertise< sensor_msgs::PointCloud2>("/clusters_ec_", 10);

    tolerance = 0.02;
    min_size = 50;
    max_size = 2000;
    threshold = 0.05;
    percent = 0.4;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tumgo_vision3D");
    ros::NodeHandle n;
    tf::TransformListener *listener = new tf::TransformListener();
    TUMgoVision3D node(n,"xtion_depth_optical_frame", listener);
    ROS_INFO("Started TUMgo_vision3D node.");
    ros::spin();
    return 0;
}


