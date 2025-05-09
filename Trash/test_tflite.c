#include "tensorflow/lite/c/c_api.h"
#include <stdio.h>

int main() {
    TfLiteModel* model = TfLiteModelCreateFromFile("../data/models/face_detection_short_range.tflite");
    if (model == NULL) {
        printf("Failed to load model.\n");
        return 1;
    }
    printf("Model loaded successfully.\n");
    TfLiteModelDelete(model);
    return 0;
}

