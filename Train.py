import face_recognition
import json
import sys

DEFAULT_OUTPUT_FILE = "encoding.json"

def main(known_image_path):
    print(f"Processing training image: {known_image_path}")
    

    known_image = face_recognition.load_image_file(known_image_path)
    known_face_encodings = face_recognition.face_encodings(known_image)

    if not known_face_encodings:
        print("No face found in the known image!")
        return

    encoding_list = known_face_encodings[0].tolist()


    with open(DEFAULT_OUTPUT_FILE, "w") as f:
        json.dump(encoding_list, f)

    print(f"Face encoding saved to {DEFAULT_OUTPUT_FILE}.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_faces.py <known_image_path>")
        sys.exit(1)

    known_image_path = sys.argv[1]
    main(known_image_path)
