#ifndef INPAINTING_H
#define INPAINTING_H

#include "utils.h"
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>

void reconstruct(const cv::Mat& depth, const cv::Mat& fillRegion, const cv::Mat& laplacian, cv::Mat& filledDepth);

#endif
