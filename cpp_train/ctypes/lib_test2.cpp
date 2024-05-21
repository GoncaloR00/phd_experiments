#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>
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


extern "C" {
    void process_image(uint8_t* data, int rows, int cols, int channels, 
                       uint8_t* data2, int rows2, int cols2, int channels2, 
                       uint16_t* coord_array, int coord_rows, int coord_cols,
                       float_t* epipolar_array, int epipolar_rows, int epipolar_cols,
                       int window_size=5, float l_ratio = 0.8) {
        cv::Mat image1 = receive_image(data,rows,cols,channels);
        cv::imshow("Image", image1);
        cv::waitKey(0);
        std::vector<cv::Point> points = convertToPoints(coord_array, coord_rows, coord_cols);
        std::vector<cv::Vec3f> epipolars = convertToEpipolar(epipolar_array, epipolar_rows, epipolar_cols);
        std::cout << "Finished" << std::endl;
        // for(const cv::Point& point : points) {
        //     std::cout << "(" << point.x << ", " << point.y << ")\n";
        // }
        std::cout << "(" << points[0].x << ", " << points[0].y << ")\n";
        std::cout << epipolars[0] << std::endl;
        }
}
