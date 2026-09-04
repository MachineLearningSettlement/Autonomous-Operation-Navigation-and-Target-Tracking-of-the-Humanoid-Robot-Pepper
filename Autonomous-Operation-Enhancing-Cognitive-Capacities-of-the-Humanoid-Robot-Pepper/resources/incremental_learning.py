import face_recognition
import numpy as np
import cv2
import os
import google.generativeai as genai

# ==========================================================
# User introduction (can be any sentence)
# ==========================================================

input_user = input("Enter the user's introduction: ")

# ==========================================================
# Configure Gemini
# ==========================================================

genai.configure(api_key="YOUR_API_KEY")

model = genai.GenerativeModel("gemini-1.5-flash")

response_person = model.generate_content(
    f"""
    Imagine you are Pepper, a humanoid robot.
    From the following sentence:

    "{input_user}"

    Return ONLY the person's first name.
    """
)

person_name = response_person.text.strip().lower()

print("Detected name:", person_name)

# ==========================================================
# Load the corresponding reference face
# ==========================================================

output_path = os.path.join("face_recog_base", f"{person_name}.png")

if not os.path.exists(output_path):
    raise FileNotFoundError(
        f"No reference image found for '{person_name}' in face_recog_base."
    )

image_base = face_recognition.load_image_file(output_path)
encoding_base = face_recognition.face_encodings(image_base)[0]

# ==========================================================
# Load the face to recognize
# ==========================================================

test_image_path = r"C:\Users\hp\Desktop\Master IA2S\Mini Projet\pythoncode\face_recog_test\test.png"

test_image = face_recognition.load_image_file(test_image_path)

face_encodings = face_recognition.face_encodings(test_image)

# ==========================================================
# Face Recognition
# ==========================================================

if len(face_encodings) > 0:

    test_encoding = face_encodings[0]

    match = face_recognition.compare_faces(
        [encoding_base],
        test_encoding
    )

    distance = face_recognition.face_distance(
        [encoding_base],
        test_encoding
    )[0]

    if match[0]:
        print(f"The detected face matches {person_name.capitalize()} "
              f"(Distance: {distance:.2f})")
    else:
        print(f"The detected face does not match {person_name.capitalize()} "
              f"(Distance: {distance:.2f})")

else:
    print("No face detected.")
