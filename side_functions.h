#pragma

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;
using namespace aruco;

void display(Mat &img)  {
    if(img.empty()) {
        std::cout << "No image loaded" << std::endl;
        return;
    }

    imshow("Display window", img);
    waitKey(0); // Wait for a keystroke in the window
}

void resize_image(Mat &img, bool grayscale=false, float fx=0.5, float fy=0.5) {
    if (grayscale)
        cvtColor(img, img, COLOR_BGR2GRAY);

    resize(img, img, Size(),0.5,0.5);
}

Mat show_many_images(vector<Mat> &vecMat) {

    if (vecMat.empty())
        return Mat();

    Mat fst_img = vecMat[0];
    int count = vecMat.size();

    int width = 0;
    for (const Mat& img : vecMat)
        width = width+img.cols;

    std::vector<int> heights{};
    heights.reserve(vecMat.size());
    for (const Mat& img : vecMat)
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

std::vector<MarkerMapPoseTracker> new_mm_tracker(const CameraParameters& camParams) {

    auto low_pos   = MarkerMapPoseTracker();
    auto high_pose = MarkerMapPoseTracker();

    auto markerMap1 = MarkerMap("data/config/block1.yml");
    auto markerMap2 = MarkerMap("data/config/block2.yml");

    low_pos.setParams(camParams, markerMap1);
    high_pose.setParams(camParams, markerMap2);

    return { low_pos, high_pose };
}

void draw_border_axis(Mat& frame, const CameraParameters& cameraParameters, std::vector<Marker> markers) {

    // draw border for each marker in the frame
    for (auto & marker : markers) {
        //std::cout << "\n" << marker << "\n" << std::endl;
        marker.draw(frame, Scalar(0, 0, 255), 2);
    }

    // draw a 3d cube in each marker if there is 3d info
    if (cameraParameters.isValid())
        for (auto & marker : markers)
            if (marker.isPoseValid()) {
                CvDrawingUtils::draw3dAxis(frame, marker, cameraParameters);
                CvDrawingUtils::draw3dCube(frame, marker, cameraParameters);
            }
}

Mat get_box_centre(const MarkerMapPoseTracker& mmap, float center_pos[4]) {
    auto rt_mat = mmap.getRTMatrix();

    for (int i=0; i<3; i++)
        center_pos[i] = center_pos[i]/1000;

    std::cout << rt_mat.rowRange(0,3) << std::endl;
    std::cout << Mat(4, 1, CV_32FC1, center_pos) << std::endl;

    Mat result = rt_mat.rowRange(0,3) * Mat(4, 1, CV_32FC1, center_pos);

    return result;
}

Mat from_3d_to_2d(const Mat& pos, const Mat& camera_transformer) {
    Mat result = camera_transformer * pos.rowRange(0,3);

    result = result/(result.at<float>(2,0));

    return result/100;
}
