import customtkinter as ctk
import cv2
import mediapipe as mp
import tkinter as tk
import subprocess
from PIL import Image, ImageTk
from customtkinter import FontManager
from scipy.spatial.distance import cosine

import tensorflow as tf
import numpy as np

import GestureRecognition

################### INTERFACE ###################
# BUTTONS
FontManager.load_font("AppleGaramond.ttf")
gr = GestureRecognition.GestureRecognition()
none_spoofed_image = None
button_width, button_height, font_size, font_size_total = 460, 70, 48, 60
corner_radius, border_color, border_width = 8, "black", 2

def create_button(parent, text, command, fg_color, hover_color, text_color):
    return ctk.CTkButton(parent, text=text, command=command, fg_color=fg_color,
                         text_color=text_color, hover_color=hover_color, font=(
                             "Arial", font_size),
                         width=button_width, height=button_height,
                         corner_radius=corner_radius, border_color=border_color,
                         border_width=border_width)

def register_interface():
    global buttonRegister, buttonCheckIn, buttonBack, register_fill

    # Hide existing buttons
    buttonRegister.pack_forget()
    buttonCheckIn.pack_forget()
    buttonBack.pack_forget()

    # Create and show the fill-in box
    register_fill = ctk.CTkEntry(
        button_frame, width=button_width, height=button_height, corner_radius=corner_radius)
    register_fill.pack(side="left", padx=10)

    buttonRegister = create_button(button_frame, "Register",
                                   lambda: register_user(register_fill.get()),
                                   "#009946", "#007D39", "white")

    buttonRegister.pack(side="left", padx=10)

def main_interface():
    global buttonRegister, buttonCheckIn, buttonBack
    
    if not verified:
        status_label.configure(text="For verification, kindly perform the following gesture response:")
    
    buttonRegister = create_button(
        button_frame, "Register", register_interface, "#009946", "#007D39", "white")
    buttonRegister.pack(side="left", padx=10)

    buttonCheckIn = create_button(
        button_frame, "Check-in", check_in, "#FF8A00", "#D97500", "white")
    buttonCheckIn.pack(side="left", padx=10)

    buttonBack = create_button(
        button_frame, "Back", back, "#F22C17", "#BF1200", "white")
    buttonBack.pack(side="left", padx=10)


def update_total_students_label():
    total_students_label.configure(
        text=f"Total Students: {len(checked_in_students)}")


def back():
    subprocess.Popen(["python", "app.py"])
    app.quit()

################### VIDEO ###################

# ANTI-SPOOFING
verified = False

def update_frame():
    global none_spoofed_image, verified
    ret, frame = cap.read()
    if ret:
        # Process frame with MediaPipe or other processing here
        image = gr.load_image_from_file(frame)

        gesture_detected = gr.recognize_hand_gesture(image)
        print(str(gr.gesture_challenge_list) + " | " +
              str(gr.skip_gesture_list) + " | " + str(gesture_detected))

        gr.update_challenge_state(gesture_detected)
        
        if not verified:
            challenge_label.configure(text="Challenge: " + str(gr.gesture_challenge_list))

        if gr.check_challenge_complete():
            gr.verified_image = frame
            gr.reset_challenge()

            # update verified label
            verified_label.configure(text="AntiSpoofed: True")
            verified = True
            challenge_label.configure(text="")
            status_label.configure(text="")
            main_interface()
            # verified_label.configure(fg_color="#009946")

            print("Challenge Complete")

        # Convert the frame to PhotoImage
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        video_label.after(10, update_frame)  # Continue updating the frame


################### SUPERVISED MODEL ###################
RECOGNITION_THRESHOLD = 0.3

embedding_model = tf.keras.models.load_model('supervised_embedding.h5')

user_embeddings = {}


def preprocess_image(image):
    image = cv2.resize(image, (375, 375))  # Resize image
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return np.expand_dims(image, axis=0)


def generate_embedding(image):
    preprocessed_image = preprocess_image(image)
    return embedding_model.predict(preprocessed_image)[0]


def register_user(user_id):
    global status_label, buttonRegister, register_fill, verified

    try:
        # ret, frame = cap.read()
        # if not ret:
        #     status_label.configure(text="Error capturing frame from camera.")
        #     return

        frame = gr.access_verified_image()

        if (frame is None):
            status_label.configure(text="Please complete the challenge first.")
            return

        embedding = generate_embedding(frame)
        user_embeddings[user_id] = embedding
        status_label.configure(text=f"User {user_id} registered successfully.")

        verified_label.configure(text="AntiSpoofed: False")
        verified_label.configure(fg_color="#FF0000")
        verified = False

        main_interface()
    except Exception as e:
        status_label.configure(text=f"Error during registration: {str(e)}")
        print(f"Error during registration: {str(e)}")


def check_in():
    global status_label, checked_in_students

    try:
        # ret, frame = cap.read()
        # if not ret:
        #     status_label.configure(text="Error capturing frame from camera.")
        #     return

        frame = gr.access_verified_image()

        if (frame is None):
            status_label.configure(text="Please complete the challenge first.")
            return

        new_embedding = generate_embedding(frame)
        min_distance = float('inf')
        recognized_user_id = "Unknown"

        for user_id, embedding in user_embeddings.items():
            distance = cosine(new_embedding, embedding)
            if distance < min_distance:
                min_distance = distance
                recognized_user_id = user_id

        # Update status label based on recognition result
        if min_distance <= RECOGNITION_THRESHOLD:
            checked_in_students.add(recognized_user_id)
            update_total_students_label()
            status_label.configure(
                text=f"Recognized User: {recognized_user_id}")
        else:
            status_label.configure(text="User not recognized.")

        verified_label.configure(text="AntiSpoofed: False")
        verified_label.configure(fg_color="#FF0000")

    except Exception as e:
        status_label.configure(text=f"Error during recognition: {str(e)}")


################### APP IMPLEMENTATION ###################
app = ctk.CTk()
app.title("Face Recognition Attendance System")
app.geometry("1440x810")
ctk.set_appearance_mode("light")

# VIDEO
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()
cap = cv2.VideoCapture(0)
main_frame = ctk.CTkFrame(app, fg_color='#FFF8E9')
main_frame.pack(expand=True, fill='both')
video_label = tk.Label(main_frame, width=930, height=635)
video_label.grid(row=0, column=0, padx=10, columnspan=2)


# TOTAL STUDENT
checked_in_students = set()

total_students_label = ctk.CTkLabel(main_frame, text=f"Total Students: {len(checked_in_students)}", font=("Apple Garamond", font_size_total))
total_students_label.grid(row=0, column=2, sticky="nw", padx=10)

verified_label = ctk.CTkLabel(main_frame, text="AntiSpoofed: False", font=("Apple Garamond", font_size_total))
verified_label.grid(row=0, column=2, sticky="sw", padx=10)

# STATUS
status_label = ctk.CTkLabel(main_frame, text="", font=("Arial", 30))
status_label.grid(row=1, column=0, padx=10, columnspan=2)

challenge_label = ctk.CTkLabel(main_frame, text="", font=("Arial", 15))
challenge_label.grid(row=2, column=1, sticky="sw", padx=10)

# BUTTON FRAME
button_frame = ctk.CTkFrame(main_frame, fg_color='#FFF8E9')
button_frame.grid(row=3, column=0, columnspan=3, pady=20)

main_interface()


################### LOOP ###################
update_frame()
app.mainloop()
cap.release()
