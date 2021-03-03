#pragma

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;

void display(Mat &img)  {
    if(img.empty()) {
        std::cout << "No image loaded" << std::endl;
        return;
    }

    imshow("Display window", img);
    waitKey(0); // Wait for a keystroke in the window
}

Mat resize_image(Mat &img, bool grayscale=false, float fx=0.5, float fy=0.5) {
    if (grayscale)
        cvtColor(img, img, COLOR_BGR2GRAY);

    resize(img, img, Size(),0.5,0.5);
    return img;
}

Mat show_many_images(std::vector<cv::Mat> &vecMat) {

    if (vecMat.empty())
        return Mat();

    Mat fst_img = vecMat[0];
    int count = vecMat.size();

    int width = 0;
    for (const Mat& img : vecMat)
        width = width+img.cols;

    std::vector<int> heights{};
    heights.reserve(vecMat.size());
    for (const Mat& img : vecMat )
            heights.push_back(img.rows);

    int height = *std::max_element(heights.begin(), heights.end());

    Mat collage(
            Size(width, height),
            fst_img.type(),
            Scalar::all(0)
    );

    int x_offset = 0;
    int y_offset = 0;

    for (const auto& img : vecMat) {
        int w = img.cols; int h = img.rows;

        img.copyTo(collage.rowRange(y_offset,y_offset+h)
                              .colRange(x_offset,x_offset+w));

        x_offset += w;
        y_offset += 0;
    }

    return collage;
}
