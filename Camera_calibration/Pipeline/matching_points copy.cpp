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
    for(int i = 1; i < rows*cols; ++i) {
        points.push_back(cv::Point(array[i-1], array[i]));
    }
    return points;
}

std::vector<cv::Vec3f> convertToEpipolar(float_t* array, int rows, int cols) {
    std::vector<cv::Vec3f> epipolars;
    for(int i = 2; i < rows*cols; ++i) {
        epipolars.push_back(cv::Vec3f(array[i-2], array[i-1], array[i]));
    }
    return epipolars;
}


std::pair<std::vector<cv::Point>, std::vector<cv::Point>> find_matching_points(cv::Mat img1, cv::Mat img2, std::vector<cv::Point> coord_array, std::vector<cv::Vec3f> epiline_array, int window_size=5, float l_ratio = 0.8) {
    std::vector<cv::Point> points_1;
    std::vector<cv::Point> points_2;
    for (int idx = 0; idx < coord_array.size(); idx++) {
        cv::Point coord = coord_array[idx];
        cv::Vec3f line = epiline_array[idx];
        float a = line[0], b = line[1], c = line[2];
        cv::Point best_match;
        float best_distance = std::numeric_limits<float>::infinity();
        float second_dist = std::numeric_limits<float>::infinity();
        for (int x = 0; x < img2.cols; x++) {
            if (b != 0) {
                int y = -1*(a*x + c) / b;
                if (y < 0 || y >= img2.rows) continue;
                cv::Rect roi1(coord.x-window_size, coord.y-window_size, 2*window_size, 2*window_size);
                cv::Rect roi2(x-window_size, y-window_size, 2*window_size, 2*window_size);
                if (roi1.x < 0 || roi1.y < 0 || roi2.x < 0 || roi2.y < 0 || roi1.x+roi1.width > img1.cols || roi1.y+roi1.height > img1.rows || roi2.x+roi2.width > img2.cols || roi2.y+roi2.height > img2.rows) continue;
                cv::Mat patch1 = img1(roi1);
                cv::Mat patch2 = img2(roi2);
                double distance = cv::norm(patch1, patch2, cv::NORM_L1);
                if (distance < best_distance) {
                    second_dist = best_distance;
                    best_distance = distance;
                    best_match = cv::Point(x, y);
                }
            }
        }
        if (second_dist == std::numeric_limits<float>::infinity() || best_distance >= l_ratio*second_dist) {
            best_match = cv::Point(-1, -1);
        }
        if (best_match.x != -1 && best_match.y != -1) {
            points_1.push_back(coord);
            points_2.push_back(best_match);
        }
    }
    // for(const cv::Point& point : points_2) {
    //         std::cout << "(" << point.x << ", " << point.y << ")\n";
    //     }
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
                       float_t* epipolar_array, int epipolar_rows, int epipolar_cols,
                       int window_size=5, float l_ratio = 0.8) {

        std::cout << "Converting from c-types to OpenCV...." << std::endl;
        cv::Mat image1 = receive_image(data,rows,cols,channels);
        std::cout << "image1" << std::endl;
        cv::Mat image2 = receive_image(data2,rows2,cols2,channels2);
        std::cout << "image2" << std::endl;
        std::vector<cv::Point> points = convertToPoints(coord_array, coord_rows, coord_cols);
        std::cout << "points" << std::endl;
        std::vector<cv::Vec3f> epipolars = convertToEpipolar(epipolar_array, epipolar_rows, epipolar_cols);
        std::cout << "epipolars" << std::endl;
        std::cout << "Conversion completed\nStarting matching points...." << std::endl;

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
    return result;
        }
}