#include <iostream>

#include "side_functions.h"

using namespace cv;

int main() {

    std::string video_path = samples::findFile("data/video/baumer_video0043.avi");
    std::string img1 = samples::findFile("data/images/img1.jpg");
    std::string img2 = samples::findFile("data/images/img2.jpg");
    Mat vid;

    Mat image1 = imread(img1, IMREAD_COLOR);
    Mat image2 = imread(img2, IMREAD_COLOR);
    Mat image3 = imread(img1, IMREAD_COLOR);

    std::vector imgVec = { image1, image2, image3};

    Mat images = show_many_images(imgVec);

    display(images);

    return 0;
}
