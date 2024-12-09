from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth, firestore

cred_path = os.path.join("static", "js", "key.json")
cred = credentials.Certificate(cred_path)

firebase_admin.initialize_app(cred)

db = firestore.client()  # Initialize Firestore


app = Flask(__name__)

# Global variables
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_trained = False
label_dict = {}
identified_names = []  # List to hold the names of identified people
tracking_active = False  # Flag to control the attendance tracking state

def train_model():
    global model_trained, label_dict
    faces = []
    labels = []
    current_label = 0

    # First pass to create label dictionary
    for filename in os.listdir('dataImage'):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter for image files
            name = filename.split('_')[0].lower()
            if name not in label_dict:
                label_dict[name] = current_label
                current_label += 1

    print("Found people:", list(label_dict.keys()))

    # Second pass to collect faces and labels
    for filename in os.listdir('dataImage'):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join('dataImage', filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                continue

            detected_faces = face_detector.detectMultiScale(image, 1.1, 5)

            if len(detected_faces) > 0:
                for (x, y, w, h) in detected_faces:
                    face = cv2.resize(image[y:y+h, x:x+w], (150, 150))
                    name = filename.split('_')[0].lower()
                    label = label_dict[name]
                    faces.append(face)
                    labels.append(label)

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        model_trained = True
        print("Model training completed")

def generate_frames():
    global identified_names  # Use the global list to store names
    global tracking_active    # Control tracking state
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.1, 5)

        if tracking_active:
            identified_names = []

        for (x, y, w, h) in faces:
            if model_trained:
                face_roi = cv2.resize(gray[y:y+h, x:x+w], (150, 150))

                try:
                    label, confidence = recognizer.predict(face_roi)

                    name = [k for k, v in label_dict.items() if v == label][0].title()

                    if confidence < 100:  # Adjust this threshold as needed
                        color = (0, 255, 0)  # Green
                        if tracking_active:
                            identified_names.append(name)  # Add identified name to the list
                    else:
                        name = "Unknown"
                        color = (0, 0, 255)  # Red

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{name} ({confidence:.1f})",
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                except Exception as e:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, "Training Required",
                          (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        status_text = "Model Trained" if model_trained else "Model Not Trained"
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if model_trained else (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route for LU Smart LOG in.html
@app.route('/', methods=['GET', 'POST'], endpoint='login')
def index():
    if request.method == 'POST':
        email = request.form.get('emailInput')
        password = request.form.get('password')

        # Process the login here
        try:
            user = auth.get_user_by_email(email)
            # Perform authentication
            return redirect(url_for('dashboard'))  # Redirect to dashboard if successful
        except auth.UserNotFoundError:
            return render_template('LU Smart LOG in.html', error_message="User not found.")
        except Exception as e:
            return render_template('LU Smart LOG in.html', error_message="An error occurred.")

    # If GET request, render the login page
    return render_template('LU Smart LOG in.html')


@app.route('/nav')
def nav():
    return render_template('nav.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        first_name = request.form.get('firstName')
        last_name = request.form.get('lastName')
        email = request.form.get('emailInput')
        password = request.form.get('password')


        try:
            user = auth.create_user(email=email, password=password)
            db.collection('lecturers').document(user.uid).set({
                'firstName': first_name,
                'lastName': last_name,
                'email': email
            })
            return redirect(url_for('dashboard'))
        except auth.EmailAlreadyExistsError:
            # If the email is already in use, pass the error message to the template
            return render_template('registration_page.html', error_message="Email is already in use. Please try another email.")
        except Exception as e:
            print(f"Error: {e}")
            # In case of any other unexpected errors, pass the error message to the template
            return render_template('registration_page.html', error_message="An unexpected error occurred during registration.")


    return render_template('registration_page.html')

# Route for forgetpass.html
@app.route('/forget_password', methods=['GET', 'POST'])
def forget_password():
    if request.method == 'POST':
        email = request.form.get('email')
        try:
            # Trigger password reset email through Firebase
            link = auth.generate_password_reset_link(email)
            # Here, you would normally send the link to the user's email
            return f"Password reset link has been sent to {email}. Please check your inbox."
        except Exception as e:
            return f"Error: {str(e)}"

    # If it's a GET request, simply render the form
    return render_template('forgetpass.html')



@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')


# Route for webcam.html
@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

# Route for after_webcam.html
@app.route('/after_webcam')
def after_webcam():
    return render_template('after_webcam.html')




# Route for Manage Course attendance.html
@app.route('/manage_course_attendance')
def manage_course_attendance():
    return render_template('Manage Course attendance.html')



# Route for select course attendance.html
@app.route('/select_course_attendance')
def select_course_attendance():
    return render_template('select course attendance.html')

# Route for update attendance before.html
@app.route('/update_attendance_before')
def update_attendance_before():
    return render_template('update attendance before.html')

# Route for update attendance final.html
@app.route('/update_attendance_final')
def update_attendance_final():
    return render_template('update attendance final.html')


# Route for view attendance.html
@app.route('/view_attendance', methods=['GET', 'POST'])
def view_attendance():
    if request.method == 'POST':
        # Get the selected subject and date from the form submission
        subject_name = request.form.get('subject')
        attendance_date = request.form.get('attendanceDate')

        if subject_name and attendance_date:
            # Render the template with the selected subject and date
            return render_template(
                'view attendance.html',
                subject=subject_name,
                date=attendance_date
            )
        else:
            # Handle case where subject or date is missing
            return "Subject or date not selected", 400

    # Default GET request
    return render_template('view attendance.html', subject="Default Course Name", date="No Date Provided")



# Route for view course attendance.html
@app.route('/view_course_attendance')
def view_course_attendance():
    return render_template('view course attendance.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/train')
def train():
    train_model()
    return "Training completed"

@app.route('/get_identified_names')
def get_identified_names():
    return jsonify(identified_names)

@app.route('/start_tracking')
def start_tracking():
    global tracking_active
    tracking_active = True
    return jsonify({"status": "Tracking started"})

@app.route('/stop_tracking')
def stop_tracking():
    global tracking_active
    tracking_active = False
    return jsonify({"status": "Tracking stopped"})


print("Starting initial training...")
train_model()
port = int(os.environ.get('PORT', 5000))  # Render sets the PORT environment variable
app.run(host='0.0.0.0', port=port)

