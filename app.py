from flask import Flask, render_template, Response, jsonify, request, redirect, url_for
import cv2
import os
import numpy as np
import firebase_admin
from firebase_admin import credentials, auth, firestore
from datetime import datetime
import time


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

# Reference to the students collection
students_collection = db.collection('students')
students_attendance = db.collection('attendances')
courses_collection = db.collection('courses')


def train_model():
    global model_trained, label_dict
    faces = []
    labels = []
    current_label = 0

    label_dict = {}  # Reset label dictionary

    # First pass: Create label dictionary based on LU IDs
    for filename in os.listdir('dataImage'):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter for image files
            lu_id = filename.split('_')[0]  # Extract LU ID
            if lu_id not in label_dict:
                label_dict[lu_id] = current_label
                current_label += 1

    print("Found LU IDs:", list(label_dict.keys()))

    # Second pass: Collect faces and labels
    for filename in os.listdir('dataImage'):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter for image files
            image_path = os.path.join('dataImage', filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Could not read image: {filename}")
                continue

            # Improve face detection by adjusting parameters
            detected_faces = face_detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in detected_faces:
                face = cv2.resize(image[y:y+h, x:x+w], (150, 150))

                # Optionally, apply histogram equalization to improve image quality
                face = cv2.equalizeHist(face)
                face = cv2.GaussianBlur(face, (5, 5), 0)

                # Map LU ID to label
                lu_id = filename.split('_')[0]
                label = label_dict[lu_id]

                faces.append(face)
                labels.append(label)

    if len(faces) > 0:
        # Train the model (increase iterations by using more data and possibly changing classifier)
        recognizer.train(faces, np.array(labels))
        model_trained = True
        print("Model training completed")
    else:
        print("No faces found for training.")


def generate_frames():
    global identified_names, tracking_active
    global students_collection  # Ensure this is accessible
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use face detection to find faces
        faces = face_detector.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        if tracking_active:
            identified_names = []  # Reset identified names when tracking is active

        for (x, y, w, h) in faces:
            if model_trained:
                face_roi = cv2.resize(gray[y:y+h, x:x+w], (150, 150))

                try:
                    label, confidence = recognizer.predict(face_roi)

                    if confidence < 1000:  # Acceptable threshold
                        lu_id = [k for k, v in label_dict.items() if v == label][0]

                        # Fetch student data from Firebase
                        student_doc = students_collection.document(lu_id).get()
                        if student_doc.exists:
                            student_data = student_doc.to_dict()
                            name = student_data.get('name', 'Unknown')
                        else:
                            name = "Unknown"

                        color = (0, 255, 0)  # Green for recognized

                        # Get the current time in AM/PM format
                        current_time = datetime.now().strftime("%I:%M:%S %p")  # 12-hour format with AM/PM

                        if tracking_active:
                            identified_names.append({'name': name, 'lu_id': lu_id, 'time': current_time})
                    else:
                        name = "Unknown"
                        lu_id = "N/A"
                        color = (0, 0, 255)  # Red for unrecognized
                        current_time = "N/A"

                    # Draw bounding box and label with time
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{name} ({lu_id})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                except Exception as e:
                    print(f"Error during prediction: {e}")
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                cv2.putText(frame, "Training Required", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Display training status
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
    # Extract the selected subject name from query parameters
    subject = request.args.get("subject")
    return render_template('webcam.html', subject=subject)

# Route to save attendance data to Firestore
@app.route('/save_attendance', methods=['POST'])
def save_attendance():
    attendance_data = request.json.get('attendance', [])
    course_id = request.args.get("course_id", "INEN-5301-01")  # Use query parameter or default
    date = datetime.now().strftime("%Y-%m-%d")

    # Step 1: Get all students enrolled in the course
    enrolled_students_query = students_collection.stream()  # Fetch all students
    enrolled_students = []

    for student in enrolled_students_query:
        student_data = student.to_dict()
        if course_id in student_data.get("enrolledCourses", []):  # Check if enrolled in the course
            enrolled_students.append({
                "lu_id": student.id,  # LU ID from document ID
                "name": student_data.get("name"),
            })

    # Step 2: Extract LU IDs of students marked as present
    present_lu_ids = {student.get("lu_id") for student in attendance_data}

    # Step 3: Save attendance records for all students in the course
    for student in enrolled_students:
        lu_id = student['lu_id']
        name = student['name']

        # Check if a record for the student already exists for the date
        existing_record_query = students_attendance.where('lu_id', '==', lu_id).where('courseId', '==', course_id).where('date', '==', date).get()

        if not existing_record_query:  # If no existing record
            # Determine the attendance status
            status = "present" if lu_id in present_lu_ids else "absent"

            # Create and save the attendance record
            attendance_record = {
                "lu_id": lu_id,
                "name": name,
                "courseId": course_id,
                "date": date,
                "status": status,
            }
            students_attendance.add(attendance_record)

    return jsonify({"status": "success", "message": "Attendance saved successfully"}), 200


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
    # Get the subject ID from the query parameter
    subject_code = request.args.get('subject')

    # Fetch course details using the subject code from Firestore
    course_doc = courses_collection.document(subject_code).get()

    # Check if the course exists
    if course_doc.exists:
        course_data = course_doc.to_dict()
        subject_name = course_data.get('name', 'Unknown Course')
    else:
        subject_name = "Course Not Found"

    # Pass the course details to the template
    return render_template('update attendance before.html', subject_code=subject_code, subject_name=subject_name)


# Route for update attendance final.html
@app.route('/update_attendance_final')
def update_attendance_final():
    return render_template('update attendance final.html')


# Route for view attendance.html
@app.route('/view_attendance', methods=['GET', 'POST'])
def view_attendance():
    if request.method == 'POST':
        subject_code = request.form.get('subject')
        attendance_date = request.form.get('attendanceDate')

        if subject_code and attendance_date:
            # Fetch the course data using the subject code
            subject_data = courses_collection.document(subject_code).get()

            # Check if the subject document exists
            if subject_data.exists:
                subject_dict = subject_data.to_dict()
                subject_name = subject_dict.get('name', 'Unknown Course')
                subject_code = subject_dict.get('code', 'Unknown Code')
                subject_description = subject_dict.get('description', 'No Description Available')
            else:
                return "Course not found", 404  # If the course doesn't exist in Firestore

            # Fetch attendance data using course code and date
            attendance_query = students_attendance.where('courseId', '==', subject_code).where('date', '==', attendance_date)
            attendance_docs = attendance_query.stream()

            attendance_data = []
            total_students = 0
            present_students = 0

            for doc in attendance_docs:
                student_data = doc.to_dict()
                lu_id = student_data.get('lu_id', 'Unknown LU ID')
                status = student_data.get('status', 'Absent').capitalize()

                # Count the present students
                if status == 'Present':
                    present_students += 1

                total_students += 1

                # Fetch student's name
                student_doc = students_collection.document(lu_id).get()
                student_name = 'Unknown Student'  # Default name in case of missing data
                if student_doc.exists:
                    student_name = student_doc.to_dict().get('name', 'Unknown Student')

                attendance_data.append({
                    'name': student_name,
                    'lu_id': lu_id,
                    'status': status
                })

            # Calculate the attendance percentage
            if total_students > 0:
                attendance_percentage = (present_students / total_students) * 100
            else:
                attendance_percentage = 0  # Prevent division by zero

            # Render the view attendance page with the attendance percentage
            return render_template('view attendance.html',
                                   subject_name=subject_name,
                                   subject_code=subject_code,
                                   subject_description=subject_description,
                                   date=attendance_date,
                                   attendance_data=attendance_data,
                                   attendance_percentage=attendance_percentage)

        return "Subject or date not selected", 400

    return render_template('view attendance.html',
                           subject_name="Default Course Name",
                           subject_code="No Code Provided",
                           subject_description="No Description Provided",
                           date="No Date Provided",
                           attendance_percentage=0)


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
    global identified_names
    try:
        # Ensure the identified names are in the format [{'name': ..., 'lu_id': ...}]
        return jsonify(identified_names)
    except Exception as e:
        print(f"Error fetching identified names: {e}")
        return jsonify([])

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



if __name__ == "__main__":
    print("Starting initial training...")
    train_model()
    port = int(os.environ.get('PORT', 5000))  # Render sets the PORT environment variable
    app.run(host='0.0.0.0', port=port)
