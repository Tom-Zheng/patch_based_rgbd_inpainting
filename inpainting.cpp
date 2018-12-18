// Image + Depth Inpainting by Tian Zheng
#include "inpainting.h"

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
typedef Eigen::Triplet<double> T;

using namespace cv;
using Eigen::MatrixXd;

// function [filledDepth] = reconstruct(depth, fillRegion, Dx, Dy)
// fillRegion : 0 for source
void reconstruct(const Mat& depth, const Mat& fillRegion, const Mat& laplacian, Mat& filledDepth)  {
    // TODO: 1. Sparse Building    2. Sparse Solving
    CV_Assert(fillRegion.depth() == CV_8U);
    CV_Assert(depth.depth() == CV_32F);
    CV_Assert(laplacian.depth() == CV_32F);
    int W = depth.cols;  // size of the image
    int H = depth.rows;
    int N = countNonZero(fillRegion);
    // Assembly: Ax = b
    std::vector<T> coefficients;            // list of non-zeros coefficients
    Eigen::VectorXd b(N);                   // the right hand side-vector resulting from the constraints
    
    //---------------- Building the problem -----------------
    coefficients.reserve(5*N);

    // build the lookup table;
    Mat lut = Mat::zeros(H, W, CV_32SC1);
    int index = 0;
    for( int i = 0; i < H; ++i)
        for( int j = 0; j < W; ++j )    {
            if (fillRegion.at<uchar>(i,j) == 0)
                continue;
            lut.at<int>(i,j) = index;
            index++;
        }
    
    // Builing A and b
    index = 0;
    for( int i = 0; i < H; ++i)
        for( int j = 0; j < W; ++j )    {
            if (fillRegion.at<uchar>(i,j) == 0)
                continue;
            float v_ij = 0;
            b[index] = laplacian.at<float>(i,j);
            if ( i >= 1)    {
                if ( fillRegion.at<uchar>(i-1,j) == 0 ) {    // Neighbour is in Source region
                    b[index] -= depth.at<float>(i-1, j);
                    v_ij -= 1;
                }
                else {                                       // Neighbour is in fill region
                    v_ij -= 1;
                    coefficients.push_back(T(index, lut.at<int>(i-1,j), 1));
                }
            }
            if ( i <= H - 2)    {
                if ( fillRegion.at<uchar>(i+1,j) == 0 ) {    // Neighbour is in Source region
                    b[index] -= depth.at<float>(i+1, j);
                    v_ij -= 1;
                }
                else {                                       // Neighbour is in fill region
                    v_ij -= 1;
                    coefficients.push_back(T(index, lut.at<int>(i+1,j), 1));
                }
            }
            if ( j >= 1)    {
                if ( fillRegion.at<uchar>(i,j-1) == 0 ) {    // Neighbour is in Source region
                    b[index] -= depth.at<float>(i, j-1);
                    v_ij -= 1;
                }
                else {                                       // Neighbour is in fill region
                    v_ij -= 1;
                    coefficients.push_back(T(index, lut.at<int>(i,j-1), 1));
                }
            }
            if ( j <= W - 2)    {
                if ( fillRegion.at<uchar>(i,j+1) == 0 ) {    // Neighbour is in Source region
                    b[index] -= depth.at<float>(i, j+1);
                    v_ij -= 1;
                }
                else {                                       // Neighbour is in fill region
                    v_ij -= 1;
                    coefficients.push_back(T(index, lut.at<int>(i,j+1), 1));
                }
            }
            coefficients.push_back(T(index, lut.at<int>(i,j), v_ij));
            index++;
        }
    
    SpMat A(N,N);
    A.setFromTriplets(coefficients.begin(), coefficients.end());
    
    // Debug: check A and b
    // std::cout << "A = " << std::endl;
    // std::cout << MatrixXd(A) << std::endl;
    // std::cout << "b = " << std::endl;
    // std::cout << b << std::endl;

    // Solve the system
    // Solving:
    Eigen::SimplicialCholesky<SpMat> solver(A);  // performs a Cholesky factorization of A
    Eigen::VectorXd x = solver.solve(b);         // use the factorization to solve for the given right hand side
    
    // Debug show x
    // std::cout << "x = " << std::endl;
    // std::cout << x << std::endl;
    
    // Filling depth
    filledDepth = depth.clone();
    index = 0;
    for( int i = 0; i < H; ++i)
        for( int j = 0; j < W; ++j )    {
            if (fillRegion.at<uchar>(i,j) == 0)
                continue;
            filledDepth.at<float>(i,j) = x[index];
            index++;
        }
    return;
}