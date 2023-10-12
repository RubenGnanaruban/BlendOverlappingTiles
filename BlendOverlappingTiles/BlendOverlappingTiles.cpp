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
    int n_X_tiles = 10;// 19;// 15;// 10;
    int n_Y_tiles = 26;// 21;// 15;// 16;

    //String dirname = "images_to_crop"; //".\\images_to_crop";
    //string dirname = "D:\\PaigeAI subtype 254\\FIBI Breast\\111\\Breast HVS-21-111 Serial H&E FIBI EDOF scan 1\\converted_olt";
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
                    

                    /*
                    if (row < col) {
                        if (col + row < imgwidth) {
                            // use top_tile
                            value += img_top.at<Vec3b>(row + M, col) * kernal_comp[row][col];
                        }
                        else {
                            // use right_tile
                            value += img_right.at<Vec3b>(row, col - N) * kernal_comp[row][col];
                        }
                    }
                    else {
                        if (col + row < imgwidth) {
                            // use left_tile
                            value += img_left.at<Vec3b>(row, col + N) * kernal_comp[row][col];
                        }
                        else {
                            // use bottom_tile
                            value += img_bottom.at<Vec3b>(row - M, col) * kernal_comp[row][col];
                        }
                    }*/

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

    /*
    Vec3b value;
    for (int row = 0; row < imgheight; row++) {
        for (int col = 0; col < imgwidth; col++) {
            //Vec3b as typename
            value = img.at<Vec3b>(row, col);
            value *= kernal_mid[row][col];
            img.at<Vec3b>(row, col) = value;
        }
    }
    imwrite(dir_blended_tiles + "\\" + "test1.jpg", img);

    int i_y = 0, i_x;
    Mat overlapping_tile;

    //Case 1: vertically aligned with the original tiles, horizontally in between tiles
    while (i_y < n_Y_tiles) {
        i_x = 0;
        while (i_x < n_X_tiles - 1) {
            int i_first_image = n_X_tiles * i_y + i_x;
            Mat img_l = imread(fn[i_first_image]);
            Mat tile_l = img_l(Range(0, imgheight), Range(N - overlap_original_sides, imgwidth - overlap_original_sides));
            Mat img_r = imread(fn[i_first_image + 1]);
            Mat tile_r = img_r(Range(0, imgheight), Range(overlap_original_sides, N + overlap_original_sides));
            cv::hconcat(tile_l, tile_r, overlapping_tile);
            imwrite(dir_blended_tiles + "\\" + getFileName(fn[i_first_image]) + '_' + ".jpg", overlapping_tile);
            ++i_x;
        }
        ++i_y;
    }
    //Case 2 and 3: vertically in between the original tiles
    i_y = 0;
    while (i_y < n_Y_tiles - 1) {
        i_x = 0;
        while (i_x < 2 * n_X_tiles - 1) {
            if (i_x % 2) {//Case 3: vertically and horizontally in between tiles
                int i_first_image = n_X_tiles * i_y + (i_x - 1) / 2;
                Mat img_top_l = imread(fn[i_first_image]);
                Mat tile_top_l = img_top_l(Range(M - overlap_original_top_bottom, imgheight - overlap_original_top_bottom), Range(N - overlap_original_sides, imgwidth - overlap_original_sides));
                Mat img_b_l = imread(fn[i_first_image + n_X_tiles]);
                Mat tile_b_l = img_b_l(Range(overlap_original_top_bottom, M + overlap_original_top_bottom), Range(N - overlap_original_sides, imgwidth - overlap_original_sides));
                Mat ol_tile_l, ol_tile_r;
                cv::vconcat(tile_top_l, tile_b_l, ol_tile_l);
                Mat img_top_r = imread(fn[i_first_image + 1]);
                Mat tile_top_r = img_top_r(Range(M - overlap_original_top_bottom, imgheight - overlap_original_top_bottom), Range(overlap_original_sides, N + overlap_original_sides));
                Mat img_b_r = imread(fn[i_first_image + n_X_tiles + 1]);
                Mat tile_b_r = img_b_r(Range(overlap_original_top_bottom, M + overlap_original_top_bottom), Range(overlap_original_sides, N + overlap_original_sides));
                cv::vconcat(tile_top_r, tile_b_r, ol_tile_r);
                cv::hconcat(ol_tile_l, ol_tile_r, overlapping_tile);
                imwrite(dir_blended_tiles + "\\" + insert_FileName(fn[i_first_image]) + '_' + ".jpg", overlapping_tile);
            }
            else {//Case 2: vertically in between, horizontally aligned
                int i_first_image = n_X_tiles * i_y + i_x / 2;
                Mat img_top = imread(fn[i_first_image]);
                Mat tile_top = img_top(Range(M - overlap_original_top_bottom, imgheight - overlap_original_top_bottom), Range(0, imgwidth));
                Mat img_b = imread(fn[i_first_image + n_X_tiles]);
                Mat tile_b = img_b(Range(overlap_original_top_bottom, M + overlap_original_top_bottom), Range(0, imgwidth));
                cv::vconcat(tile_top, tile_b, overlapping_tile);
                imwrite(dir_blended_tiles + "\\" + insert_FileName(fn[i_first_image]) + ".jpg", overlapping_tile);
            }
            ++i_x;
        }
        ++i_y;
    }*/
}

