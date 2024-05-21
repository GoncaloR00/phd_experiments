#include <iostream>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>

extern "C" {
    void process_image(uint8_t* data, int rows, int cols, int channels, uint8_t* data2, int rows2, int cols2, int channels2) {
        cv::Mat image(rows, cols, channels == 1 ? CV_8UC1 : CV_8UC3, data);
        cv::Mat image2(rows2, cols2, channels2 == 1 ? CV_8UC1 : CV_8UC3, data2);
        cv::imshow("Image", image);
        cv::imshow("Image2", image2);
        cv::waitKey(0);

    }
}
