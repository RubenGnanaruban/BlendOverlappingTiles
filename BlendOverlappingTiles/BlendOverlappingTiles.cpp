/*
* Author: Vimalatharmaiyah Gnanaruban
* October 2022
*/

#include <string>
#include <iostream>
#include<opencv2/opencv.hpp>
//#include <fstream>
//#include <cstdlib>
#include <filesystem>
#include <opencv2/core/utility.hpp>
#include <vector>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
#define OPEN_CV_VERSION 343 //version 4.6.0

string getFileName(const string& s) {
    char sep = '\\';
    size_t i = s.rfind(sep, s.length());
    if (i != string::npos)
    {
        string filename = s.substr(i + 1, s.length() - i);
        size_t lastindex = filename.find_last_of(".");
        string rawname = filename.substr(0, lastindex);
        return(rawname);
    }
    return("");
}

string makeSiblingFolder(string dir_raw, string new_folder_name) {
  size_t ii = dir_raw.rfind('\\', dir_raw.length());
  string dir_patch = dir_raw.substr(0, ii);
  dir_patch += "\\";
  dir_patch += new_folder_name;
  //dir_patch += "_" + dir_raw.substr(ii + 1, dir_raw.length());
  fs::create_directories(dir_patch);
  return dir_patch;
}

int main()
{
    int imgheight, imgwidth, M, N;
    int N_x = 2;
    int N_y = 2;
    int n_X_tiles = 10; // Number of horizontal tiles
    int n_Y_tiles = 26; // Number of vertical tiles

    //string dirname = "D:\\PaigeAI subtype 254\\FIBI Breast\\Breast HVS-21-120 Serial H&E FIBI EDOF scan 1\\converted_olt";
    //string dirname = "D:\\PaigeAI subtype 254\\FIBI Breast\\Breast HVS-21-123 section 2 Serial H&E FIBI EDOF scan 1\\Breast HVS-21-123 section 2 Serial H&E FIBI EDOF scan 1\\converted_olt"; 
    string dirname = "C:\\Users\\Histo\\Documents\\ProstateTissue_for_CycleGAN\\Tanishq\\191";
    //string dir_blended_tiles = dirname + "\\blended_tiles";
    string dir_blended_tiles = makeSiblingFolder(dirname, "blended_tiles");

    vector<cv::String> fn;
    glob(dirname, fn, false);

    fs::create_directories(dir_blended_tiles);
    size_t count = fn.size(); //number of jpg files in original tiles
#if OPEN_CV_VERSION < 300
    Mat img = imread(fn[0], CV_LOAD_IMAGE_COLOR);
#else
    Mat img = imread(fn[0], cv::IMREAD_COLOR);
#endif
    imgheight = img.rows;
    imgwidth = img.cols;
    M = imgheight / N_y;
    N = imgwidth / N_x;

    /* The following is just to convert images to jpg*/
    /*
    int i_y = 0, i_x;
    while (i_y < n_Y_tiles) {
      i_x = 0;
      while (i_x < n_X_tiles) {
        int i_mid_tile = n_X_tiles * i_y + i_x;
        Mat img_mid = imread(fn[i_mid_tile], cv::IMREAD_COLOR);
        //convert image with the same name:
        //imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_mid_tile]) + ".jpg", img_mid);
        std::ostringstream ssy, ssx;
        ssy << std::setw(5) << std::setfill('0') << i_y;
        std::string stry = ssy.str();
        ssx << std::setw(5) << std::setfill('0') << i_x;
        std::string strx = ssx.str();
        imwrite(dir_blended_tiles + "\\Y" + stry + " X" + strx + ".jpg", img_mid);
        ++i_x;
      }
      ++i_y;
    }*/


    vector<vector<double>> kernal_mid(imgheight, vector<double>(imgwidth));
    vector<vector<double>> kernal_comp(imgheight, vector<double>(imgwidth));
    for (int row = 0; row < imgheight; row++) {
        for (int col = 0; col < imgwidth; col++) {
            //Vec3b as typename
            double pyramid_l = (double)col / N;
            double pyramid_t = (double)row / M;
            double pyramid_r = (double)(imgwidth - col) / N;
            double pyramid_b = (double)(imgheight - row) / M;
            kernal_mid[row][col] = min(min(pyramid_l, pyramid_t), min(pyramid_r, pyramid_b));
            kernal_comp[row][col] = 1 - kernal_mid[row][col];
        }
    }

    int i_y = 1, i_x;
    Vec3b value;
    Mat blended_tile;
    while (i_y < n_Y_tiles - 1) {
        i_x = 1;
        while (i_x < n_X_tiles - 1) {
            int i_mid_tile = (3 * n_X_tiles - 1) * i_y + 2 * i_x; //these are the original tiles
#if OPEN_CV_VERSION < 300
            Mat img_mid = imread(fn[i_mid_tile], CV_LOAD_IMAGE_COLOR);
            Mat img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], CV_LOAD_IMAGE_COLOR);
            Mat img_left = imread(fn[i_mid_tile - 1], CV_LOAD_IMAGE_COLOR);
            Mat img_right = imread(fn[i_mid_tile + 1], CV_LOAD_IMAGE_COLOR);
            Mat img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], CV_LOAD_IMAGE_COLOR);
#else
            Mat img_mid = imread(fn[i_mid_tile], cv::IMREAD_COLOR);
            Mat img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], cv::IMREAD_COLOR);
            Mat img_left = imread(fn[i_mid_tile - 1], cv::IMREAD_COLOR);
            Mat img_right = imread(fn[i_mid_tile + 1], cv::IMREAD_COLOR);
            Mat img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], cv::IMREAD_COLOR);
#endif
            for (int row = 0; row < imgheight; row++) {
                for (int col = 0; col < imgwidth; col++) {
                    value = img_mid.at<Vec3b>(row, col);
                    value *= kernal_mid[row][col];
                    if (row < M) {//top
                        if (col < N) {//left
                            double top_vs_left = (double)col / (row + col);
                            value += (img_top.at<Vec3b>(row + M, col) * top_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - top_vs_left))* kernal_comp[row][col];
                        }
                        else {//right
                            double top_vs_right = (double)(imgwidth - col) / (row + imgwidth - col);
                            value += (img_top.at<Vec3b>(row + M, col) * top_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - top_vs_right)) * kernal_comp[row][col];
                        }
                    }
                    else {//bottom
                        if (col < N) {//left
                            double bottom_vs_left = (double)col / (imgheight - row + col);
                            value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - bottom_vs_left)) * kernal_comp[row][col];
                        }
                        else {//right
                            double bottom_vs_right = (double)(imgwidth - col) / (imgheight - row + imgwidth - col);
                            value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - bottom_vs_right)) * kernal_comp[row][col];
                        }
                    }

                    img_mid.at<Vec3b>(row, col) = value;
                }
            }
            imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_mid_tile]) + ".jpg", img_mid);
            ++i_x;
        }
        ++i_y;
    }
    
    /*Top row :
     */
    i_y = 0, i_x = 1;
    while (i_x < n_X_tiles - 1) {
        int i_mid_tile = (3 * n_X_tiles - 1) * i_y + 2 * i_x; //these are the original tiles
#if OPEN_CV_VERSION < 300
        Mat img_mid = imread(fn[i_mid_tile], CV_LOAD_IMAGE_COLOR);
        Mat img_left = imread(fn[i_mid_tile - 1], CV_LOAD_IMAGE_COLOR);
        Mat img_right = imread(fn[i_mid_tile + 1], CV_LOAD_IMAGE_COLOR);
        Mat img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], CV_LOAD_IMAGE_COLOR);
#else
        Mat img_mid = imread(fn[i_mid_tile], cv::IMREAD_COLOR);
        Mat img_left = imread(fn[i_mid_tile - 1], cv::IMREAD_COLOR);
        Mat img_right = imread(fn[i_mid_tile + 1], cv::IMREAD_COLOR);
        Mat img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], cv::IMREAD_COLOR);
#endif
        Vec3b value;
        for (int row = 0; row < imgheight; row++) {
            for (int col = 0; col < imgwidth; col++) {
                value = img_mid.at<Vec3b>(row, col);
                value *= kernal_mid[row][col];
                if (row < M) {//top : use the mid tile
                    if (col < N) {//left
                        double top_vs_left = (double)(N - col) / N;
                        value += (img_mid.at<Vec3b>(row, col) * top_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - top_vs_left)) * kernal_comp[row][col];
                    }
                    else {//right
                        double top_vs_right = (double)(col - N) / N;
                        value += (img_mid.at<Vec3b>(row, col) * top_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - top_vs_right)) * kernal_comp[row][col];
                    }
                }
                else {//bottom
                    if (col < N) {//left
                        double bottom_vs_left = (double)col / (imgheight - row + col);
                        value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - bottom_vs_left)) * kernal_comp[row][col];
                    }
                    else {//right
                        double bottom_vs_right = (double)(imgwidth - col) / (imgheight - row + imgwidth - col);
                        value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - bottom_vs_right)) * kernal_comp[row][col];
                    }
                }
                img_mid.at<Vec3b>(row, col) = value;
            }
        }
        imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_mid_tile]) + ".jpg", img_mid);
        ++i_x;
    }

    /*Bottom row :
     */
    i_y = n_Y_tiles - 1, i_x = 1;
    while (i_x < n_X_tiles - 1) {
        int i_mid_tile = (3 * n_X_tiles - 1) * i_y + 2 * i_x; //these are the original tiles
#if OPEN_CV_VERSION < 300
        Mat img_mid = imread(fn[i_mid_tile], CV_LOAD_IMAGE_COLOR);
        Mat img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], CV_LOAD_IMAGE_COLOR);
        Mat img_left = imread(fn[i_mid_tile - 1], CV_LOAD_IMAGE_COLOR);
        Mat img_right = imread(fn[i_mid_tile + 1], CV_LOAD_IMAGE_COLOR);
#else
        Mat img_mid = imread(fn[i_mid_tile], cv::IMREAD_COLOR);
        Mat img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], cv::IMREAD_COLOR);
        Mat img_left = imread(fn[i_mid_tile - 1], cv::IMREAD_COLOR);
        Mat img_right = imread(fn[i_mid_tile + 1], cv::IMREAD_COLOR);
#endif
        Vec3b value;
        for (int row = 0; row < imgheight; row++) {
            for (int col = 0; col < imgwidth; col++) {
                value = img_mid.at<Vec3b>(row, col);
                value *= kernal_mid[row][col];
                if (row < M) {//top
                    if (col < N) {//left
                        double top_vs_left = (double)col / (row + col);
                        value += (img_top.at<Vec3b>(row + M, col) * top_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - top_vs_left)) * kernal_comp[row][col];
                    }
                    else {//right
                        double top_vs_right = (double)(imgwidth - col) / (row + imgwidth - col);
                        value += (img_top.at<Vec3b>(row + M, col) * top_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - top_vs_right)) * kernal_comp[row][col];
                    }
                }
                else {//bottom
                    if (col < N) {//left
                        double bottom_vs_left = (double)(N - col) / N;
                        value += (img_mid.at<Vec3b>(row, col) * bottom_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - bottom_vs_left)) * kernal_comp[row][col];
                    }
                    else {//right
                        double bottom_vs_right = (double)(col - N) / N;
                        value += (img_mid.at<Vec3b>(row, col) * bottom_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - bottom_vs_right)) * kernal_comp[row][col];
                    }
                }
                img_mid.at<Vec3b>(row, col) = value;
            }
        }
        imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_mid_tile]) + ".jpg", img_mid);
        ++i_x;
    }

    /*Left column
     */
    i_y = 1, i_x = 0;
    while (i_y < n_Y_tiles - 1) {
        int i_mid_tile = (3 * n_X_tiles - 1) * i_y + 2 * i_x; //these are the original tiles
#if OPEN_CV_VERSION < 300
        Mat img_mid = imread(fn[i_mid_tile], CV_LOAD_IMAGE_COLOR);
        Mat img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], CV_LOAD_IMAGE_COLOR);
        Mat img_right = imread(fn[i_mid_tile + 1], CV_LOAD_IMAGE_COLOR);
        Mat img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], CV_LOAD_IMAGE_COLOR);
#else
        Mat img_mid = imread(fn[i_mid_tile], cv::IMREAD_COLOR);
        Mat img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], cv::IMREAD_COLOR);
        Mat img_right = imread(fn[i_mid_tile + 1], cv::IMREAD_COLOR);
        Mat img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], cv::IMREAD_COLOR);
#endif
        Vec3b value;
        for (int row = 0; row < imgheight; row++) {
            for (int col = 0; col < imgwidth; col++) {
                value = img_mid.at<Vec3b>(row, col);
                value *= kernal_mid[row][col];
                if (row < M) {//top
                    if (col < N) {//left
                        double top_vs_left = (double)(M - row) / M;
                        value += (img_top.at<Vec3b>(row + M, col) * top_vs_left + img_mid.at<Vec3b>(row, col) * (1 - top_vs_left)) * kernal_comp[row][col];
                    }
                    else {//right
                        double top_vs_right = (double)(imgwidth - col) / (row + imgwidth - col);
                        value += (img_top.at<Vec3b>(row + M, col) * top_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - top_vs_right)) * kernal_comp[row][col];
                    }
                }
                else {//bottom
                    if (col < N) {//left
                        double bottom_vs_left = (double)(row - M) / M;
                        value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_left + img_mid.at<Vec3b>(row, col) * (1 - bottom_vs_left)) * kernal_comp[row][col];
                    }
                    else {//right
                        double bottom_vs_right = (double)(imgwidth - col) / (imgheight - row + imgwidth - col);
                        value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - bottom_vs_right)) * kernal_comp[row][col];
                    }
                }
                img_mid.at<Vec3b>(row, col) = value;
            }
        }
        imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_mid_tile]) + ".jpg", img_mid);
        ++i_y;
    }

    /*Right column
     */
    i_y = 1, i_x = n_X_tiles - 1;
    while (i_y < n_Y_tiles - 1) {
        int i_mid_tile = (3 * n_X_tiles - 1) * i_y + 2 * i_x; //these are the original tiles
#if OPEN_CV_VERSION < 300
        Mat img_mid = imread(fn[i_mid_tile], CV_LOAD_IMAGE_COLOR);
        Mat img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], CV_LOAD_IMAGE_COLOR);
        Mat img_left = imread(fn[i_mid_tile - 1], CV_LOAD_IMAGE_COLOR);
        Mat img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], CV_LOAD_IMAGE_COLOR);
#else
        Mat img_mid = imread(fn[i_mid_tile], cv::IMREAD_COLOR);
        Mat img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], cv::IMREAD_COLOR);
        Mat img_left = imread(fn[i_mid_tile - 1], cv::IMREAD_COLOR);
        Mat img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], cv::IMREAD_COLOR);
#endif
        Vec3b value;
        for (int row = 0; row < imgheight; row++) {
            for (int col = 0; col < imgwidth; col++) {
                value = img_mid.at<Vec3b>(row, col);
                value *= kernal_mid[row][col];
                if (row < M) {//top
                    if (col < N) {//left
                        double top_vs_left = (double)col / (row + col);
                        value += (img_top.at<Vec3b>(row + M, col) * top_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - top_vs_left)) * kernal_comp[row][col];
                    }
                    else {//right
                        double top_vs_right = (double)(M - row) / M;
                        value += (img_top.at<Vec3b>(row + M, col) * top_vs_right + img_mid.at<Vec3b>(row, col) * (1 - top_vs_right)) * kernal_comp[row][col];
                    }
                }
                else {//bottom
                    if (col < N) {//left
                        double bottom_vs_left = (double)col / (imgheight - row + col);
                        value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - bottom_vs_left)) * kernal_comp[row][col];
                    }
                    else {//right
                        //double bottom_vs_right = (double)(row - M) / (imgheight - row + imgwidth - col);
                        double bottom_vs_right = (double)(row - M) / M;
                        value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_right + img_mid.at<Vec3b>(row, col) * (1 - bottom_vs_right)) * kernal_comp[row][col];
                    }
                }
                img_mid.at<Vec3b>(row, col) = value;
            }
        }
        imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_mid_tile]) + ".jpg", img_mid);
        ++i_y;
    }

    /*Top left corner tile
     */
    i_y = 0, i_x = 0;
    int i_mid_tile = (3 * n_X_tiles - 1) * i_y + 2 * i_x; //these are the original tiles
#if OPEN_CV_VERSION < 300
    Mat img_mid = imread(fn[i_mid_tile], CV_LOAD_IMAGE_COLOR);
    Mat img_right = imread(fn[i_mid_tile + 1], CV_LOAD_IMAGE_COLOR);
    Mat img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], CV_LOAD_IMAGE_COLOR);
#else
    Mat img_mid = imread(fn[i_mid_tile], cv::IMREAD_COLOR);
    Mat img_right = imread(fn[i_mid_tile + 1], cv::IMREAD_COLOR);
    Mat img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], cv::IMREAD_COLOR);
#endif
    value;
    for (int row = 0; row < imgheight; row++) {
        for (int col = 0; col < imgwidth; col++) {
            value = img_mid.at<Vec3b>(row, col);
            value *= kernal_mid[row][col];
            if (row < M) {//top
                if (col < N) {//left
                    value += img_mid.at<Vec3b>(row, col) * kernal_comp[row][col];
                }
                else {//right
                    double top_vs_right = (double)(imgwidth - col) / N;
                    value += (img_mid.at<Vec3b>(row, col) * top_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - top_vs_right)) * kernal_comp[row][col];
                }
            }
            else {//bottom
                if (col < N) {//left
                    double bottom_vs_left = (double)(row - M) / M;
                    value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_left + img_mid.at<Vec3b>(row, col) * (1 - bottom_vs_left)) * kernal_comp[row][col];
                }
                else {//right
                    double bottom_vs_right = (double)(imgwidth - col) / (imgheight - row + imgwidth - col);
                    value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - bottom_vs_right)) * kernal_comp[row][col];
                }
            }
            img_mid.at<Vec3b>(row, col) = value;
        }
    }
    imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_mid_tile]) + ".jpg", img_mid);

    /*Top right corner tile
     */
    i_y = 0, i_x = n_X_tiles - 1;
    i_mid_tile = (3 * n_X_tiles - 1) * i_y + 2 * i_x; //these are the original tiles
#if OPEN_CV_VERSION < 300
    img_mid = imread(fn[i_mid_tile], CV_LOAD_IMAGE_COLOR);
    Mat img_left = imread(fn[i_mid_tile - 1], CV_LOAD_IMAGE_COLOR);
    img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], CV_LOAD_IMAGE_COLOR);
#else
    img_mid = imread(fn[i_mid_tile], cv::IMREAD_COLOR);
    Mat img_left = imread(fn[i_mid_tile - 1], cv::IMREAD_COLOR);
    img_bottom = imread(fn[i_mid_tile + 2 * n_X_tiles - i_x - 1], cv::IMREAD_COLOR);
#endif
    for (int row = 0; row < imgheight; row++) {
        for (int col = 0; col < imgwidth; col++) {
            value = img_mid.at<Vec3b>(row, col);
            value *= kernal_mid[row][col];
            if (row < M) {//top
                if (col < N) {//left
                    double top_vs_left = (double)col / N;
                    value += (img_mid.at<Vec3b>(row, col) * top_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - top_vs_left)) * kernal_comp[row][col];
                }
                else {//right
                    value += img_mid.at<Vec3b>(row, col) * kernal_comp[row][col];
                }
            }
            else {//bottom
                if (col < N) {//left
                    double bottom_vs_left = (double)col / (imgheight - row + col);
                    value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - bottom_vs_left)) * kernal_comp[row][col];
                }
                else {//right
                    double bottom_vs_right = (double)(row - M) / M;
                    value += (img_bottom.at<Vec3b>(row - M, col) * bottom_vs_right + img_mid.at<Vec3b>(row, col) * (1 - bottom_vs_right)) * kernal_comp[row][col];
                }
            }
            img_mid.at<Vec3b>(row, col) = value;
        }
    }
    imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_mid_tile]) + ".jpg", img_mid);

    /*Bottom left corner tile
     */
    i_y = n_Y_tiles - 1, i_x = 0;
    i_mid_tile = (3 * n_X_tiles - 1) * i_y + 2 * i_x; //these are the original tiles
#if OPEN_CV_VERSION < 300
    img_mid = imread(fn[i_mid_tile], CV_LOAD_IMAGE_COLOR);
    Mat img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], CV_LOAD_IMAGE_COLOR);
    img_right = imread(fn[i_mid_tile + 1], CV_LOAD_IMAGE_COLOR);
#else
    img_mid = imread(fn[i_mid_tile], cv::IMREAD_COLOR);
    Mat img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], cv::IMREAD_COLOR);
    img_right = imread(fn[i_mid_tile + 1], cv::IMREAD_COLOR);
#endif
    for (int row = 0; row < imgheight; row++) {
        for (int col = 0; col < imgwidth; col++) {
            value = img_mid.at<Vec3b>(row, col);
            value *= kernal_mid[row][col];
            if (row < M) {//top
                if (col < N) {//left
                    double top_vs_left = (double)(M - row) / M; //corrected!
                    value += (img_top.at<Vec3b>(row + M, col) * top_vs_left + img_mid.at<Vec3b>(row, col) * (1 - top_vs_left)) * kernal_comp[row][col];
                }
                else {//right
                    double top_vs_right = (double)(imgwidth - col) / (row + imgwidth - col);
                    value += (img_top.at<Vec3b>(row + M, col) * top_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - top_vs_right)) * kernal_comp[row][col];
                }
            }
            else {//bottom
                if (col < N) {//left
                    value += img_mid.at<Vec3b>(row, col) * kernal_comp[row][col];
                }
                else {//right
                    double bottom_vs_right = (double)(imgwidth - col) / N;
                    value += (img_mid.at<Vec3b>(row, col) * bottom_vs_right + img_right.at<Vec3b>(row, col - N) * (1 - bottom_vs_right)) * kernal_comp[row][col];
                }
            }
            img_mid.at<Vec3b>(row, col) = value;
        }
    }
    imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_mid_tile]) + ".jpg", img_mid);

    /*Bottom right corner tile
     */
    i_y = n_Y_tiles - 1, i_x = n_X_tiles - 1;
    i_mid_tile = (3 * n_X_tiles - 1) * i_y + 2 * i_x; //these are the original tiles
#if OPEN_CV_VERSION < 300
    img_mid = imread(fn[i_mid_tile], CV_LOAD_IMAGE_COLOR);
    img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], CV_LOAD_IMAGE_COLOR);
    img_left = imread(fn[i_mid_tile - 1], CV_LOAD_IMAGE_COLOR);
#else
    img_mid = imread(fn[i_mid_tile], cv::IMREAD_COLOR);
    img_top = imread(fn[i_mid_tile - n_X_tiles - i_x], cv::IMREAD_COLOR);
    img_left = imread(fn[i_mid_tile - 1], cv::IMREAD_COLOR);
#endif
    for (int row = 0; row < imgheight; row++) {
        for (int col = 0; col < imgwidth; col++) {
            value = img_mid.at<Vec3b>(row, col);
            value *= kernal_mid[row][col];
            if (row < M) {//top
                if (col < N) {//left
                    double top_vs_left = (double)col / (row + col);
                    value += (img_top.at<Vec3b>(row + M, col) * top_vs_left + img_left.at<Vec3b>(row, col + N) * (1 - top_vs_left)) * kernal_comp[row][col];
                }
                else {//right
                    double top_vs_right = (double)(M - row) / M;
                    value += (img_top.at<Vec3b>(row + M, col) * top_vs_right + img_mid.at<Vec3b>(row, col) * (1 - top_vs_right)) * kernal_comp[row][col];
                }
            }
            else {//bottom
                if (col < N) {//left
                    double bottom_vs_left = (double)col / N;
                    value += (img_mid.at<Vec3b>(row - M, col) * bottom_vs_left + img_left.at<Vec3b>(row, col) * (1 - bottom_vs_left)) * kernal_comp[row][col];
                }
                else {//right
                    value += img_mid.at<Vec3b>(row, col) * kernal_comp[row][col];
                }
            }
            img_mid.at<Vec3b>(row, col) = value;
        }
    }
    imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_mid_tile]) + ".jpg", img_mid);

}

