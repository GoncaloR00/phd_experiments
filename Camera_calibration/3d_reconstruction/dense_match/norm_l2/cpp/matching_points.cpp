#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>


cv::Mat receive_image(uint8_t* data, int rows, int cols, int channels){
    cv::Mat image = cv::Mat (rows, cols, channels == 1 ? CV_8UC1 : CV_8UC3, data);
    return image;
}

std::vector<cv::Point> convertToPoints(uint16_t* array, int rows, int cols) {
    std::vector<cv::Point> points;
    for(int i = 0; i < rows*cols; i+=2) {
        points.push_back(cv::Point(array[i], array[i+1]));
    }
    return points;
}

std::vector<cv::Vec3d> convertToEpipolar(double_t* array, int rows, int cols) {
    std::vector<cv::Vec3d> epipolars;

    for(int i = 0; i < rows; i++) {
        epipolars.push_back(cv::Vec3d(array[i], array[i+rows], array[i+(2*rows)]));
    }
    return epipolars;
}


#include <opencv2/opencv.hpp>
#include <vector>

std::pair<std::vector<cv::Point>, std::vector<cv::Point>> find_matching_points(cv::Mat img1, cv::Mat img2, std::vector<cv::Point> coord_array, std::vector<cv::Vec3d> epiline_array, int window_size=5, float l_ratio = 0.8, int normType=cv::NORM_L2) {
    std::vector<cv::Point> points_1;
    std::vector<cv::Point> points_2;

    for (int idx = 0; idx < coord_array.size(); idx++) {
        cv::Point best_match;
        float second_dist = std::numeric_limits<float>::infinity();
        float best_distance = std::numeric_limits<float>::infinity();

        double a = epiline_array[idx][0];
        double b = epiline_array[idx][1];
        double c = epiline_array[idx][2];

        for (int x = 0; x < img2.cols; x++) {
            if (b != 0) {
                int y = -1*(a*x + c) / b;
                if (y < 0 || y >= img2.rows) continue;

                int a1 = coord_array[idx].y - window_size;
                int b1 = coord_array[idx].y + window_size;
                int c1 = coord_array[idx].x - window_size;
                int d1 = coord_array[idx].x + window_size;
                int a2 = y - window_size;
                int b2 = y + window_size;
                int c2 = x - window_size;
                int d2 = x + window_size;

                if (a1 < 0 || c1 < 0 || a2 < 0 || c2 < 0 || b1 > img1.rows || d1 > img1.cols || b2 > img2.rows || d2 > img2.cols) continue;

                cv::Mat patch1 = img1(cv::Rect(c1, a1, window_size*2, window_size*2));
                cv::Mat patch2 = img2(cv::Rect(c2, a2, window_size*2, window_size*2));


                double distance = cv::norm(patch1, patch2, normType);

                if (distance < best_distance) {
                    second_dist = best_distance;
                    best_distance = distance;
                    best_match = cv::Point(x, y);
                }
            }
        }

        if (second_dist == std::numeric_limits<float>::infinity() || best_distance >= l_ratio * second_dist) {
            best_match = cv::Point(-1, -1);
        }

        if (best_match.x != -1 && best_match.y != -1) {
            points_1.push_back(coord_array[idx]);
            points_2.push_back(best_match);
        }
    }
    return std::make_pair(points_1, points_2);
}

extern "C" {
    struct Point {
        int x;
        int y;
    };

    struct PointsPair {
        Point* first;
        int first_size;
        Point* second;
        int second_size;
    };
}


extern "C" {
    PointsPair process_data(uint8_t* data, int rows, int cols, int channels, 
                       uint8_t* data2, int rows2, int cols2, int channels2, 
                       uint16_t* coord_array, int coord_rows, int coord_cols,
                       double_t* epipolar_array, int epipolar_rows, int epipolar_cols,
                       int window_size=5, float l_ratio = 0.8) {

        std::cout << "########### Converting from c-types to OpenCV C++ ###########" << std::endl;
        cv::Mat image1 = receive_image(data,rows,cols,channels);
        std::cout << "[ OK ] Image1 converted         | Size: " << image1.size() << std::endl;
        cv::Mat image2 = receive_image(data2,rows2,cols2,channels2);
        std::cout << "[ OK ] Image2 converted         | Size: " << image2.size() << std::endl;
        std::vector<cv::Point> points = convertToPoints(coord_array, coord_rows, coord_cols);
        std::cout << "[ OK ] Coordinates converted    | Size: " << points.size() << " rows" << std::endl;
        std::vector<cv::Vec3d> epipolars = convertToEpipolar(epipolar_array, epipolar_rows, epipolar_cols);
        std::cout << "[ OK ] Epipolar lines converted | Size: " << epipolars.size() << " rows" << std::endl;
        std::cout << "################### Conversion completed ####################\nStarting matching points...." << std::endl;

        std::pair<std::vector<cv::Point>, std::vector<cv::Point>> matches = find_matching_points(image1, image2, points, epipolars, window_size, l_ratio);

        PointsPair result;
        result.first_size = matches.first.size();
    result.first = new Point[result.first_size];
    for (int i = 0; i < result.first_size; ++i) {
        result.first[i].x = matches.first[i].x;
        result.first[i].y = matches.first[i].y;
    }

    result.second_size = matches.second.size();
    result.second = new Point[result.second_size];
    for (int i = 0; i < result.second_size; ++i) {
        result.second[i].x = matches.second[i].x;
        result.second[i].y = matches.second[i].y;
    }
    std::cout << "#################### Matching completed ######################\nSending to python..." << std::endl;
    return result;
        }
}