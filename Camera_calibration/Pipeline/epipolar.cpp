#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

std::tuple<std::vector<KeyPoint>, std::vector<KeyPoint>, std::vector<DMatch>> feature_match(Mat& img1, Mat& img2, float lowe_ratio = 0.3) {
    Ptr<SIFT> detector = SIFT::create();
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    BFMatcher matcher(NORM_L2);
    std::vector<std::vector<DMatch>> knn_matches;
    matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

    std::vector<DMatch> matches;
    for(size_t i = 0; i < knn_matches.size(); i++) {
        if(knn_matches[i][0].distance < lowe_ratio * knn_matches[i][1].distance) {
            matches.push_back(knn_matches[i][0]);
        }
    }

    return std::make_tuple(keypoints1, keypoints2, matches);
}

int main() {
    Mat img1 = imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_1.png", IMREAD_GRAYSCALE);
    Mat img2 = imread("/home/gribeiro/PhD/phd_experiments/Camera_calibration/Feature_match/Sift/Images/astra/curved/img_0_2.png", IMREAD_GRAYSCALE);

    std::vector<KeyPoint> keypoints1, keypoints2;
    std::vector<DMatch> matches;
    std::tie(keypoints1, keypoints2, matches) = feature_match(img1, img2);

    // Draw matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);

    imshow("Matches", img_matches);
    waitKey();

    return 0;
}
