 #include "utils.h"

// utility functions needed for inpainting

/* 
 * Return a % b where % is the mathematical modulus operator.
 */
int mod(int a, int b) {
    return ((a % b) + b) % b;
}


/*
 * Load the color, depth, mask images with a border of size
 * radius around every image to prevent boundary collisions when taking patches
 */
void loadInpaintingImages(
                          const std::string& colorFilename,
                          const std::string& depthFilename,
                          const std::string& maskFilename,
                          cv::Mat& colorMat,
                          cv::Mat& depthMat,
                          cv::Mat& maskMat,
                          double scale)
{
    assert(colorFilename.length() && maskFilename.length() && depthFilename.length());
    
    colorMat    = cv::imread(colorFilename, cv::IMREAD_COLOR);
    depthMat = cv::imread(depthFilename, cv::IMREAD_UNCHANGED);
    maskMat     = cv::imread(maskFilename, cv::IMREAD_UNCHANGED);
    
    cv::resize(colorMat, colorMat, cv::Size(), scale, scale);
    cv::resize(depthMat, depthMat, cv::Size(), scale, scale);
    cv::resize(maskMat, maskMat, cv::Size(), scale, scale);
    
    assert(colorMat.size() == maskMat.size());
    assert(!colorMat.empty() && !maskMat.empty());
    
    // convert colorMat to depth CV_32F for colorspace conversions
    colorMat.convertTo(colorMat, CV_32F);
    colorMat /= 255.0f;

    depthMat.convertTo(depthMat, CV_32F);
    depthMat /= 255.0f;
    
    // add border around colorMat
    cv::copyMakeBorder(
                       colorMat,
                       colorMat,
                       RADIUS,
                       RADIUS,
                       RADIUS,
                       RADIUS,
                       cv::BORDER_CONSTANT,
                       cv::Scalar_<float>(0,0,0)
                       );
    cv::copyMakeBorder(
                       depthMat,
                       depthMat,
                       RADIUS,
                       RADIUS,
                       RADIUS,
                       RADIUS,
                       cv::BORDER_CONSTANT,
                       cv::Scalar_<float>(0)
                       );
}


/*
 * Show a Mat object quickly. For testing purposes only.
 */
void showMat(const cv::String& winname, const cv::Mat& mat, int time/*= 5*/)
{
    assert(!mat.empty());
    cv::namedWindow(winname);
    cv::imshow(winname, mat);
    cv::waitKey(time);
    cv::destroyWindow(winname);
}


/*
 * Extract closed boundary from mask.
 */
void getContours(const cv::Mat& mask,
                 contours_t& contours,
                 hierarchy_t& hierarchy
                 )
{
    assert(mask.type() == CV_8UC1);
    cv::findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
}


/*
 * Get a patch of size RAIDUS around point p in mat.
 */
cv::Mat getPatch(const cv::Mat& mat, const cv::Point& p)
{
    assert(RADIUS <= p.x && p.x < mat.cols-RADIUS && RADIUS <= p.y && p.y < mat.rows-RADIUS);
    return  mat(
                 cv::Range(p.y-RADIUS, p.y+RADIUS+1),
                 cv::Range(p.x-RADIUS, p.x+RADIUS+1)
                 );
}


// get the x and y derivatives of a patch centered at patchCenter in image
// computed using a 3x3 Scharr filter
void getDerivatives(const cv::Mat& grayMat, cv::Mat& dx, cv::Mat& dy)
{
    assert(grayMat.type() == CV_32FC1);
    
    cv::Sobel(grayMat, dx, -1, 1, 0, -1);
    cv::Sobel(grayMat, dy, -1, 0, 1, -1);
}


/*
 * Get the unit normal of a dense list of boundary points centered around point p.
 */
cv::Point2f getNormal(const contour_t& contour, const cv::Point& point)
{
    int sz = (int) contour.size();
    
    assert(sz != 0);
    
    int pointIndex = (int) (std::find(contour.begin(), contour.end(), point) - contour.begin());
    
    assert(pointIndex != contour.size());
    
    if (sz == 1)
    {
        return cv::Point2f(1.0f, 0.0f);
    } else if (sz < 2 * BORDER_RADIUS + 1)
    {
        // Too few points in contour to use LSTSQ regression
        // return the normal with respect to adjacent neigbourhood
        cv::Point adj = contour[(pointIndex + 1) % sz] - contour[pointIndex];
        return cv::Point2f(adj.y, -adj.x) / cv::norm(adj);
    }
    
    // Use least square regression
    // create X and Y mat to SVD
    cv::Mat X(cv::Size(2, 2*BORDER_RADIUS+1), CV_32F);
    cv::Mat Y(cv::Size(1, 2*BORDER_RADIUS+1), CV_32F);
    
    assert(X.rows == Y.rows && X.cols == 2 && Y.cols == 1 && X.type() == Y.type()
              && Y.type() == CV_32F);
    
    int i = mod((pointIndex - BORDER_RADIUS), sz);
    
    float* Xrow;
    float* Yrow;
    
    int count = 0;
    int countXequal = 0;
    while (count < 2*BORDER_RADIUS+1)
    {
        Xrow = X.ptr<float>(count);
        Xrow[0] = contour[i].x;
        Xrow[1] = 1.0f;
        
        Yrow = Y.ptr<float>(count);
        Yrow[0] = contour[i].y;
        
        if (Xrow[0] == contour[pointIndex].x)
        {
            ++countXequal;
        }
        
        i = mod(i+1, sz);
        ++count;
    }
    
    if (countXequal == count)
    {
        return cv::Point2f(1.0f, 0.0f);
    }
    // to find the line of best fit
    cv::Mat sol;
    cv::solve(X, Y, sol, cv::DECOMP_SVD);
    
    assert(sol.type() == CV_32F);
    
    float slope = sol.ptr<float>(0)[0];
    cv::Point2f normal(-slope, 1);
    
    return normal / cv::norm(normal);
}


/*
 * Return the confidence of confidencePatch
 */
double computeConfidence(const cv::Mat& confidencePatch)
{
    return cv::sum(confidencePatch)[0] / (double) confidencePatch.total();
}


/*
 * Iterate over every contour point in contours and compute the
 * priority of path centered at point using grayMat and confidenceMat
 */
void computePriority(const contours_t& contours, const cv::Mat& grayMat, const cv::Mat& confidenceMat, cv::Mat& priorityMat)
{
    assert(grayMat.type() == CV_32FC1 &&
              priorityMat.type() == CV_32FC1 &&
              confidenceMat.type() == CV_32FC1
              );
    
    // define some patches
    cv::Mat confidencePatch;
    cv::Mat magnitudePatch;
    
    cv::Point2f normal;
    cv::Point maxPoint;
    cv::Point2f gradient;
    
    double confidence;
    
    // get the derivatives and magnitude of the greyscale image
    cv::Mat dx, dy, magnitude;
    getDerivatives(grayMat, dx, dy);
    cv::magnitude(dx, dy, magnitude);
    
    // mask the magnitude
    cv::Mat maskedMagnitude(magnitude.size(), magnitude.type(), cv::Scalar_<float>(0));
    magnitude.copyTo(maskedMagnitude, (confidenceMat != 0.0f));
    cv::erode(maskedMagnitude, maskedMagnitude, cv::Mat());
    
    assert(maskedMagnitude.type() == CV_32FC1);
    
    // for each point in contour
    cv::Point point;
    
    for (int i = 0; i < contours.size(); ++i)
    {
        contour_t contour = contours[i];
        
        for (int j = 0; j < contour.size(); ++j)
        {
            
            point = contour[j];
            
            confidencePatch = getPatch(confidenceMat, point);
            
            // get confidence of patch
            confidence = cv::sum(confidencePatch)[0] / (double) confidencePatch.total();
            assert(0 <= confidence && confidence <= 1.0f);
            
            // get the normal to the border around point
            normal = getNormal(contour, point);
            
            // get the maximum gradient in source around patch
            magnitudePatch = getPatch(maskedMagnitude, point);
            cv::minMaxLoc(magnitudePatch, NULL, NULL, NULL, &maxPoint);
            gradient = cv::Point2f(
                                   -getPatch(dy, point).ptr<float>(maxPoint.y)[maxPoint.x],
                                   getPatch(dx, point).ptr<float>(maxPoint.y)[maxPoint.x]
                                 );
            
            // set the priority in priorityMat
            priorityMat.ptr<float>(point.y)[point.x] = std::abs((float) confidence * gradient.dot(normal));
            assert(priorityMat.ptr<float>(point.y)[point.x] >= 0);
        }
    }
}


/*
 * Transfer the values from patch centered at psiHatQ to patch centered at psiHatP in
 * mat according to maskMat.
 */
void transferPatch(const cv::Point& psiHatQ, const cv::Point& psiHatP, cv::Mat& mat, const cv::Mat& maskMat)
{
    assert(maskMat.type() == CV_8U);
    assert(mat.size() == maskMat.size());
    assert(RADIUS <= psiHatQ.x && psiHatQ.x < mat.cols-RADIUS && RADIUS <= psiHatQ.y && psiHatQ.y < mat.rows-RADIUS);
    assert(RADIUS <= psiHatP.x && psiHatP.x < mat.cols-RADIUS && RADIUS <= psiHatP.y && psiHatP.y < mat.rows-RADIUS);
    
    // copy contents of psiHatQ to psiHatP with mask
    getPatch(mat, psiHatQ).copyTo(getPatch(mat, psiHatP), getPatch(maskMat, psiHatP));
}

/*
 * Runs template matching with tmplate and mask tmplateMask on source.
 * Resulting Mat is stored in result.
 *
 */
cv::Mat computeSSD(const cv::Mat& tmplate, const cv::Mat& source, const cv::Mat& tmplateMask)
{
    assert(tmplate.type() == CV_32FC3 && source.type() == CV_32FC3);
    assert(tmplate.rows <= source.rows && tmplate.cols <= source.cols);
    assert(tmplateMask.size() == tmplate.size() && tmplate.type() == tmplateMask.type());
    
    cv::Mat result(source.rows - tmplate.rows + 1, source.cols - tmplate.cols + 1, CV_32F, 0.0f);
    
    cv::matchTemplate(source,
                      tmplate,
                      result,
                      CV_TM_SQDIFF,
                      tmplateMask
                      );
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX);
    cv::copyMakeBorder(result, result, RADIUS, RADIUS, RADIUS, RADIUS, cv::BORDER_CONSTANT, 1.1f);
    
    return result;
}

void computeGradient(const cv::Mat& src, cv::Mat& dx, cv::Mat& dy)   {
    cv::Mat kernelx = (cv::Mat_<float>(1,3)<<-0.5, 0, 0.5);
    cv::Mat kernely = (cv::Mat_<float>(3,1)<<-0.5, 0, 0.5);
    cv::filter2D(src, dx, CV_32F, kernelx, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    cv::filter2D(src, dy, CV_32F, kernely, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
    dx.col(dx.cols - 1) *= 2;
    dx.col(0) *= 2;
    dy.row(dy.rows - 1) *= 2;
    dy.row(0) *= 2;
}

void computeLaplacian(const cv::Mat& src, cv::Mat& laplacian) {
    cv::Mat src_blur;
    int kernel_size = 1;
    double scale = 1.0;
    double delta = 0;
    int border = cv::BORDER_REPLICATE;
    // Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
    GaussianBlur( src, src_blur, cv::Size(3, 3), 0, 0, border);
    cv::Laplacian( src_blur, laplacian, CV_32F, kernel_size, scale, delta, border);
}

void printMat(const cv::Mat& src, std::string name)   {
    std::cout << name << " = " << std::endl << cv::format(src, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
}