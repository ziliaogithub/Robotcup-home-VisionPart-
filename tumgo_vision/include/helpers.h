/* ***************************************
	Author:	Karinne Ramirez-Amaro
	E-mail:	karinne.ramirez@tum.de

*/


#ifndef HELPERS_H
#define HELPERS_H

#endif // HELPERS_H

// Standard libraries
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

std::string IntToString(int a);
std::string FloatToString(float a);
int StringToInt(string a);
float StringToFloat(string a);

// Helper functions
/**
 * Compares contours regarding area sizes
 * @param contour_1 first compared contour, vector<cv::Point> type,
 * @param contour_2 second compared contour, vector<cv::Point> type.
 */
bool compare_area(vector<cv::Point> contour_1,vector<cv::Point> contour_2)
{
    // rotated rectangle
    cv::RotatedRect boundRect1 = cv::minAreaRect(cv::Mat(contour_1));
    cv::RotatedRect boundRect2 = cv::minAreaRect(cv::Mat(contour_2));
    // calculate area
    float area_1 = boundRect1.size.width*boundRect1.size.height;
    float area_2 = boundRect2.size.width*boundRect2.size.height;

    return area_1 < area_2;
}
/**
 * Compares contours regarding sizes
 * @param contour_1 first compared contour, vector<cv::Point> type,
 * @param contour_2 second compared contour, vector<cv::Point> type.
 */
bool compare_size(vector<cv::Point> contour_1,vector<cv::Point> contour_2)
{
  return contour_1.size() < contour_2.size();
}
/**
 * Compares contours regarding y coordinates of the center
 * @param contour_1 first compared contour, vector<cv::Point> type,
 * @param contour_2 second compared contour, vector<cv::Point> type.
 */
bool compare_y(vector<cv::Point> contour_1,vector<cv::Point> contour_2)
{
    // rotated rectangle
    cv::RotatedRect boundRect1 = cv::minAreaRect(cv::Mat(contour_1));
    cv::RotatedRect boundRect2 = cv::minAreaRect(cv::Mat(contour_2));
    // calculate area
    float point_1 = boundRect1.center.y;
    float point_2 = boundRect2.center.y;

    return point_1 < point_2;
}


/**
 * Compares contours regarding x coordinates of the center
 * @param contour_1 first compared contour, vector<cv::Point> type,
 * @param contour_2 second compared contour, vector<cv::Point> type.
 */
bool compare_x(vector<cv::Point> contour_1,vector<cv::Point> contour_2)
{
    // rotated rectangle
    cv::RotatedRect boundRect1 = cv::minAreaRect(cv::Mat(contour_1));
    cv::RotatedRect boundRect2 = cv::minAreaRect(cv::Mat(contour_2));
    // calculate area
    float point_1 = boundRect1.center.x;
    float point_2 = boundRect2.center.x;

    return point_1 < point_2;
}


