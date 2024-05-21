#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <stdio.h>

extern "C" {
    std::pair<std::vector<cv::Point>, std::vector<cv::Point>> find_matching_points(cv::Mat img1, cv::Mat img2, std::vector<cv::Point> coord_array, std::vector<cv::Vec3f> epiline_array, int window_size=5, float l_ratio = 0.8) {
    std::vector<cv::Point> points_1;
    std::vector<cv::Point> points_2;
    std::cout << coord_array << std::endl <<std::flush;
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
    return std::make_pair(points_1, points_2);
}
    int main(){
        
    }
}