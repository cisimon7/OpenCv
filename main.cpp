#include <iostream>

#include <aruco.h>
#include "block_centers.h"
#include "side_functions.h"

using namespace cv;
using namespace aruco;

int main() {

    std::string video_path = samples::findFile("data/video/baumer_video0043.avi");
    Mat frame, gray_frame;
    std::vector<Marker> markers;

    VideoCapture cap(video_path);
    bool ret = cap.read(frame);

    resize_image(frame);
    cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

    CameraParameters cameraParameters;
    cameraParameters.readFromXMLFile("data/config/camera_parameters.yml");

    MarkerDetector detector;

    // Finding frame with at least two markers, one for the lower box and another for the upper box
    MarkerMapPoseTracker low_pos, high_pos;
    while (ret) {
        detector.setDictionary("ARUCO_MIP_36h12", 0.9);
        detector.detect(frame, markers, cameraParameters);

        auto poses = new_mm_tracker(cameraParameters);
        low_pos  = poses[0];
        high_pos = poses[1];

        low_pos.estimatePose(markers);
        high_pos.estimatePose(markers);

        if (low_pos.getRTMatrix().size && high_pos.getRTMatrix().size) {
           break;
        }

        ret = cap.read(frame);
        detector.detect(frame, markers, cameraParameters);
    }

    Mat frame_copy = frame.clone();

    detector.detect(frame, markers, cameraParameters);
    draw_border_axis(frame, cameraParameters, markers);

    std::cout << low_pos.getRTMatrix() << std::endl;

//    display(frame_copy);

    auto tvec1 = get_box_centre(low_pos, block1[2]);
    auto tvec_img1 = from_3d_to_2d(tvec1, cameraParameters.CameraMatrix);

    auto rvec_img1 = Mat(3, 1, CV_64FC1, euler1);
    auto rvec_img2 = Mat(3, 1, CV_64FC1, euler2);

    std::cout<< rvec_img1 << std::endl;
    std::cout<< tvec_img1 << std::endl;

    CvDrawingUtils::draw3dAxis(frame_copy, cameraParameters, rvec_img1, tvec_img1, 5);

    vector<Mat> images = { frame, frame_copy };
    auto collage = show_many_images(images);
    display(collage);

    return 0;
}
