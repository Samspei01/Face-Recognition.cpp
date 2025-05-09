#include "FaceLib/landmark_extractor.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <dlib/dnn.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <iostream>
#include <vector>
#include <string>
#include <dlib/opencv.h>

using namespace dlib;
using namespace std;

// Define the ResNet for face recognition
template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
                           alevel0<
                           alevel1<
                           alevel2<
                           alevel3<
                           alevel4<
                           max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
                           input_rgb_image_sized<150>
                           >>>>>>>>>>>>;

// Function to extract aligned face chip
matrix<rgb_pixel> extract_aligned_face(const cv_image<bgr_pixel>& dlib_face, const full_object_detection& face_landmarks) {
    chip_details face_chip_details = get_face_chip_details(face_landmarks, 150, 0.25);
    matrix<rgb_pixel> aligned_face;
    extract_image_chip(dlib_face, face_chip_details, aligned_face);
    return aligned_face;
}

int main() try {
    // Paths to models
    string detectionModelPath = "../data/models/face_detection_short_range.tflite";
    string landmarksModelPath = "../data/models/face_landmarks.tflite";
    string shapePredictorPath = "../models/shape_predictor_68_face_landmarks.dat";
    string faceRecognitionModelPath = "../models/dlib_face_recognition_resnet_model_v1.dat";

    // Initialize LandmarkExtractor
    LandmarkExtractor extractor(detectionModelPath, landmarksModelPath);

    // Load shape predictor for landmarks
    shape_predictor sp;
    deserialize(shapePredictorPath) >> sp;

    // Load face recognition model
    anet_type net;
    deserialize(faceRecognitionModelPath) >> net;

    // Storage for user face descriptors
    std::vector<matrix<float, 0, 1>> user_descriptors;

    // Process and store descriptors for 5 user photos
    for (int i = 1; i <= 5; ++i) {
        string imagePath = "../data_test/test" + to_string(i) + ".jpeg";
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            cerr << "Error: Unable to load image: " << imagePath << endl;
            continue;
        }

        try {
            cv_image<bgr_pixel> dlib_face(image);

            // Detect faces
            frontal_face_detector detector = get_frontal_face_detector();
            std::vector<rectangle> faces = detector(dlib_face);

            if (faces.empty()) {
                cerr << "No faces detected in the image: " << imagePath << endl;
                continue;
            }

            // Loop through all detected faces
            for (size_t j = 0; j < faces.size(); ++j) {
                full_object_detection shape = sp(dlib_face, faces[j]);

                // Align the face
                matrix<rgb_pixel> aligned_face = extract_aligned_face(dlib_face, shape);

                // Generate descriptor
                matrix<float, 0, 1> face_descriptor = net(aligned_face);
                user_descriptors.push_back(face_descriptor);

                // Draw bounding box on the face
                // cv::rectangle(image,
                //               cv::Point(faces[j].left(), faces[j].top()),
                //               cv::Point(faces[j].right(), faces[j].bottom()),
                //               cv::Scalar(0, 255, 0), 2);

                // Display aligned face
                // cv::imshow("Aligned Face", dlib::toMat(aligned_face));
                // cv::waitKey(0);
            }

            //Display the detected faces
            // cv::imshow("Detected Faces", image);
            // cv::waitKey(0);
        } catch (std::exception& ex) {
            cerr << "Exception processing image " << imagePath << ": " << ex.what() << endl;
        }
    }

    if (user_descriptors.size() < 5) {
        cerr << "Insufficient user photos detected." << endl;
        return 1;
    }

    // Save descriptors
    serialize("user_descriptors.dat") << user_descriptors;
    cout << "User descriptors saved successfully!" << endl;

    return 0;
} catch (std::exception& e) {
    cerr << "Exception: " << e.what() << endl;
    return 1;
}
