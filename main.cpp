//  main.cpp
//  Unit test

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <string>

#include "utils.h"
#include "inpainting.h"

using namespace std;

#define DEBUG 0

// Parameters
double scale = 0.5;


/*
int main (int argc, char** argv) {
    // --------------- read filename strings ------------------
    std::string colorFilename, depthFilename, maskFilename;
    
    if (argc == 4)  {
        colorFilename = argv[1];
        depthFilename = argv[2];
        maskFilename = argv[3];
    }
    else  {
        std::cerr << "Usage: ./inpainting colorImageFile depthFile maskImageFile" << std::endl;
        return -1;
    }
    
    // ---------------- read the images ------------------------
    // colorMat     - color picture + border
    // maskMat      - mask picture + border
    // depthMat     - depth 
    // grayMat      - gray picture + border
    cv::Mat colorMat, depthMat, maskMat, grayMat;
    loadInpaintingImages(
                        colorFilename,
                        depthFilename,
                        maskFilename,
                        colorMat,
                        depthMat,
                        maskMat,
                        scale);
    
    cv::cvtColor(colorMat, grayMat, CV_BGR2GRAY);
    
    if (DEBUG) {
        showMat("mask", maskMat, 0);
    }
    // confidenceMat - confidence picture + border
    // maskMat type: 1 for source, 0 for mask
    cv::Mat confidenceMat;
    maskMat.convertTo(confidenceMat, CV_32F);
    confidenceMat /= 255.0f;

    // add borders around maskMat and confidenceMat
    cv::copyMakeBorder(maskMat, maskMat,
                       RADIUS, RADIUS, RADIUS, RADIUS,
                       cv::BORDER_CONSTANT, 255);
    cv::copyMakeBorder(confidenceMat, confidenceMat,
                       RADIUS, RADIUS, RADIUS, RADIUS,
                       cv::BORDER_CONSTANT, 0.0001f);
    
    // ---------------- start the algorithm -----------------
    
    contours_t contours;            // mask contours
    hierarchy_t hierarchy;          // contours hierarchy
    
    
    // priorityMat - priority values for all contour points + border
    cv::Mat priorityMat(
                        confidenceMat.size(),
                        CV_32FC1
                        );  // priority value matrix for each contour point
    
    assert(
           colorMat.size() == grayMat.size() &&
           colorMat.size() == confidenceMat.size() &&
           colorMat.size() == maskMat.size()
           );
    
    cv::Point psiHatP;          // psiHatP - point of highest confidence
    
    cv::Mat psiHatPColor;       // color patch around psiHatP
    
    cv::Mat psiHatPConfidence;  // confidence patch around psiHatP
    double confidence;          // confidence of psiHatPConfidence
    
    cv::Point psiHatQ;          // psiHatQ - point of closest patch
    
    cv::Mat result;             // holds result from template matching
    cv::Mat erodedMask;         // eroded mask
    
    cv::Mat templateMask;       // mask for template match (3 channel)
    
    // eroded mask is used to ensure that psiHatQ is not overlapping with target
    cv::erode(maskMat, erodedMask, cv::Mat(), cv::Point(-1, -1), RADIUS);
    
    cv::Mat drawMat;
    
    
    // main loop
    const size_t area = maskMat.total();
    
    while (cv::countNonZero(maskMat) != area)   // end when target is filled
    {
        // set priority matrix to -.1, lower than 0 so that border area is never selected
        priorityMat.setTo(-0.1f);
        
        // get the contours of mask
        getContours((maskMat == 0), contours, hierarchy);
        
        if (DEBUG) {
            drawMat = colorMat.clone();
        }
        
        // compute the priority for all contour points
        computePriority(contours, grayMat, confidenceMat, priorityMat);
        
        // get the patch with the greatest priority
        cv::minMaxLoc(priorityMat, NULL, NULL, NULL, &psiHatP);
        psiHatPColor = getPatch(colorMat, psiHatP);
        psiHatPConfidence = getPatch(confidenceMat, psiHatP);
        
        cv::Mat confInv = (psiHatPConfidence != 0.0f);
        confInv.convertTo(confInv, CV_32F);
        confInv /= 255.0f;
        // get the patch in source with least distance to psiHatPColor wrt source of psiHatP
        cv::Mat mergeArrays[3] = {confInv, confInv, confInv};
        cv::merge(mergeArrays, 3, templateMask);
        result = computeSSD(psiHatPColor, colorMat, templateMask);
        
        // set all target regions to 1.1, which is over the maximum value possilbe
        // from SSD
        result.setTo(1.1f, erodedMask == 0);
        // get minimum point of SSD between psiHatPColor and colorMat
        cv::minMaxLoc(result, NULL, NULL, &psiHatQ);
        
        assert(psiHatQ != psiHatP);
        
        if (DEBUG) {
            cv::rectangle(drawMat, psiHatP - cv::Point(RADIUS, RADIUS), psiHatP + cv::Point(RADIUS+1, RADIUS+1), cv::Scalar(255, 0, 0));
            cv::rectangle(drawMat, psiHatQ - cv::Point(RADIUS, RADIUS), psiHatQ + cv::Point(RADIUS+1, RADIUS+1), cv::Scalar(0, 0, 255));
            showMat("red - psiHatQ", drawMat);
        }
        // updates
        // copy from psiHatQ to psiHatP for each colorspace
        transferPatch(psiHatQ, psiHatP, grayMat, (maskMat == 0));
        transferPatch(psiHatQ, psiHatP, colorMat, (maskMat == 0));
        
        // fill in confidenceMat with confidences C(pixel) = C(psiHatP)
        confidence = computeConfidence(psiHatPConfidence);
        assert(0 <= confidence && confidence <= 1.0f);
        // update confidence
        psiHatPConfidence.setTo(confidence, (psiHatPConfidence == 0.0f));
        // update maskMat
        maskMat = (confidenceMat != 0.0f);
    }
    
    showMat("final result", colorMat, 0);
    return 0;
}
*/


// @@@@@@ Debug Testing @@@@@@@@@
int main (int argc, char** argv) {
    
    // Test 1 Gradient
    cout << "-------------- Compute Gradient --------------" << endl;
    cv::Mat A = (cv::Mat_<float>(3,3) << 1, 2, 3, 4, 5, 6, 7, 8, 9);
    printMat(A, "A");

    cv::Mat dx, dy;
    computeGradient(A, dx, dy);

    printMat(dx, "dx");
    printMat(dy, "dy");
    
    // Test 2 Laplacian
    cout << "-------------- Compute Laplacian --------------" << endl;
    cv::Mat laplacian;

    computeLaplacian(A, laplacian);
    printMat(laplacian, "laplacian");

    // Test 4 Sparse Matrix Building...
    cout << "-------------- Sparse Building --------------" << endl;
    
    cv::Mat filled;
    cv::Mat fillRegion = (cv::Mat_<uchar>(3,3) << 0, 255, 0, 255, 255, 255, 0, 255, 0);
    reconstruct(A, fillRegion, laplacian, filled);
    
    printMat(A, "A");
    printMat(fillRegion, "fillRegion");
    printMat(filled, "filled");

    return 0;
}