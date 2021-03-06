#ifndef utils_hpp
#define utils_hpp

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <assert.h>
#include <stdio.h>

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

typedef std::vector<std::vector<cv::Point>> contours_t;
typedef std::vector<cv::Vec4i> hierarchy_t;
typedef std::vector<cv::Point> contour_t;


// Patch raduius
#define RADIUS 5
// The maximum number of pixels around a specified point on the target outline
#define BORDER_RADIUS 5

int mod(int a, int b);

void loadInpaintingImages(
                          const std::string& colorFilename,
                          const std::string& depthFilename,
                          const std::string& maskFilename,
                          cv::Mat& colorMat,
                          cv::Mat& depthMat,
                          cv::Mat& maskMat,
                          double scale);

void showMat(const cv::String& winname, const cv::Mat& mat, int time=500);

void getContours(const cv::Mat& mask, contours_t& contours, hierarchy_t& hierarchy);

double computeConfidence(const cv::Mat& confidencePatch);

cv::Mat getPatch(const cv::Mat& image, const cv::Point& p);

void getDerivatives(const cv::Mat& grayMat, cv::Mat& dx, cv::Mat& dy);

cv::Point2f getNormal(const contour_t& contour, const cv::Point& point);

void computePriority(const contours_t& contours, const cv::Mat& grayMat, const cv::Mat& confidenceMat, cv::Mat& priorityMat);

void transferPatch(const cv::Point& psiHatQ, const cv::Point& psiHatP, cv::Mat& mat, const cv::Mat& maskMat);

cv::Mat computeSSD(const cv::Mat& tmplate, const cv::Mat& source, const cv::Mat& tmplateMask);

// Add by Tian Zheng
void computeGradient(const cv::Mat& src, cv::Mat& dx, cv::Mat& dy);

void computeLaplacian(const cv::Mat& src, cv::Mat& laplacian);

void printMat(const cv::Mat& src, std::string name);

#endif