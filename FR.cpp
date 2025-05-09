#include "FaceLib/landmark_extractor.h"
#include <opencv2/highgui.hpp>
#include <dlib/dnn.h>
#include <dlib/image_processing/shape_predictor.h>
#include <dlib/opencv.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <dlib/matrix.h>
using json = nlohmann::json;
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
void write_json(const string& filename, int result) {
    ofstream json_file(filename);
    if (json_file.is_open()) {
        json_file << "{\n";
        json_file << "  \"recognized\": " << result << "\n";
        json_file << "}\n";
        json_file.close();
        cout << "JSON file written to " << filename << "\n";
    } else {
        cerr << "Error: Unable to open JSON file for writing!" << "\n";
    }
}
using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
                           alevel0<
                           alevel1<
                           alevel2<
                           alevel3<
                           alevel4<
                           max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
                           input_rgb_image_sized<150>
                           >>>>>>>>>>>>;

std::vector<matrix<float, 0, 1>> load_face_encodings(const std::string& filename) {
    std::vector<matrix<float, 0, 1>> encodings;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Unable to open JSON file: " << filename << endl;
        return encodings;
    }

    json j;
    file >> j;  // Load JSON content

    // Convert JSON array to dlib matrix
    matrix<float, 0, 1> encoding;
    encoding.set_size(j.size());
    for (size_t i = 0; i < j.size(); ++i) {
        encoding(i) = j[i];
    }

    encodings.push_back(encoding);
    return encodings;
}
int main(int argc, char** argv) try {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <image_path>" << "\n";
        return 1;
    }

    //ios_base::sync_with_stdio(false); cin.tie(NULL); cout.tie(NULL);

    string imageFile = argv[1];
    string detectionModelPath = "../data/models/face_detection_short_range.tflite";
    string landmarksModelPath = "../data/models/face_landmarks.tflite";
    string shapePredictorPath = "../models/shape_predictor_68_face_landmarks.dat";

    // Load models
    LandmarkExtractor extractor(detectionModelPath, landmarksModelPath);
    shape_predictor sp;
    deserialize(shapePredictorPath) >> sp;

    anet_type net;
    deserialize("../models/dlib_face_recognition_resnet_model_v1.dat") >> net;

    std::vector<matrix<float, 0, 1>> user_descriptors = load_face_encodings("/home/samsepi0l/Project/FaceRecognition/face/mediapipe-cpp/encoding.json");


    cv::Mat image = cv::imread(imageFile);
    if (image.empty()) {
        cerr << "Error: Unable to load image!" << "\n";
        return 1;
    }

    try {
        // Ensure the image is in the correct format
        cv_image<bgr_pixel> dlib_image(image);

        // Detect faces using the LandmarkExtractor
        std::vector<cv::Point2i> landmarks = extractor.Process(image);

        if (landmarks.empty()) {
            cerr << "No landmarks detected in the image!" << "\n";
            return 1;
        }

        // Calculate bounding box
        int min_x = image.cols, min_y = image.rows;
        int max_x = 0, max_y = 0;

        for (const auto& point : landmarks) {
            min_x = std::min(min_x, point.x);
            min_y = std::min(min_y, point.y);
            max_x = std::max(max_x, point.x);
            max_y = std::max(max_y, point.y);
        }

        // Add padding to the bounding box
        int padding = 20;
        min_x = std::max(0, min_x - padding);
        min_y = std::max(0, min_y - padding);
        max_x = std::min(image.cols, max_x + padding);
        max_y = std::min(image.rows, max_y + padding);

        cv::Rect face_box(min_x, min_y, max_x - min_x, max_y - min_y);
        cv::Mat face = image(face_box);

        // Detect landmarks for the face
        cv_image<bgr_pixel> dlib_face(face);
        auto shape = sp(dlib_face, rectangle(0, 0, face.cols, face.rows));

        // Align the face
        chip_details face_chip_details = get_face_chip_details(shape, 150, 0.25);
        matrix<rgb_pixel> aligned_face;
        extract_image_chip(dlib_face, face_chip_details, aligned_face);

        // Compute face descriptor
        matrix<float, 0, 1> face_descriptor = net(aligned_face);

        // Match face descriptor with known descriptors
        bool is_user = false;
        for (const auto& user_descriptor : user_descriptors) {
            double distance = length(user_descriptor - face_descriptor);
          printf("Descriptor Distance: %lf\n",distance);
            if (distance < 0.51) {
                is_user = true;
                break;
            }
        }
        write_json("output.json", is_user ? 1 : 0);
        if(is_user)printf("Reconized\n");
        else printf("Not reconized\n");

        // Draw results
        
        cv::Scalar color = is_user ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::rectangle(image, face_box, color, 2);
        std::string label = is_user ? "User Recognized" : "Unknown";
        cv::putText(image, label, cv::Point(face_box.x, face_box.y - 10), 	  cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);

        // Show the aligned face
        cv::imshow("Aligned Face", toMat(aligned_face));
        cv::imshow("Detected Face", image);
        cv::waitKey(0);
        cv::destroyAllWindows();
        

    } catch (const std::exception& ex) {
        cerr << "Exception: " << ex.what() << "\n";
    }

    return 0;
}
catch (std::exception& e) {
    cerr << "Exception: " << e.what() << "\n";
    return 1;
}
