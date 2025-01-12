from flask import Flask, redirect, render_template, request, make_response, session, abort, jsonify, url_for, Response
import secrets
from functools import wraps
import firebase_admin
from firebase_admin import credentials, firestore, auth
from datetime import timedelta
import os
from dotenv import load_dotenv
import cv2
from ultralytics import YOLO
import base64
import numpy as np

load_dotenv()



app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY')
camera=None

model = YOLO('best.pt')

# Configure session cookie settings
app.config['SESSION_COOKIE_SECURE'] = True  # Ensure cookies are sent over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to cookies
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Adjust session expiration as needed
app.config['SESSION_REFRESH_EACH_REQUEST'] = False
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Can be 'Strict', 'Lax', or 'None'


# Firebase Admin SDK setup
cred = credentials.Certificate("firebase-auth.json")
firebase_admin.initialize_app(cred)
db = firestore.client()


translation_txt = "No gesture detected"

def generate_frames():
    global translation_txt
    global camera
    conf_threshold = 0.5


    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)

    while True:
        ## read the camera frame
        success,frame=camera.read()
        if not success:
             break

        resized_frame=cv2.resize(frame,(640,640))
        results=model.predict(source=frame,save=False,show=False)

        translation_txt = "No gesture detected"
        # for result in results:
        #     for box in result.boxes.xyxy:
        #         x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
        #         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        #         label = result.names[int(result.boxes.cls[0])]
        #         cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
        for result in results:
            for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
                if conf >= conf_threshold:  # Only process if confidence exceeds threshold
                    x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
                    label = result.names[int(result.boxes.cls[0])]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)


                    translation_txt = label

            ret,buffer=cv2.imencode('.jpg',frame)
            frame_bytes=buffer.tobytes()

            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            
        

########################################
""" Authentication and Authorization """

# Decorator for routes that require authentication
def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is authenticated
        if 'user' not in session:
            return redirect(url_for('login'))
        
        else:
            return f(*args, **kwargs)
        
    return decorated_function


@app.route('/auth', methods=['POST'])
def authorize():
    token = request.headers.get('Authorization')
    if not token or not token.startswith('Bearer '):
        return "Unauthorized", 401

    token = token[7:]  # Strip off 'Bearer ' to get the actual token

    # try:
    #     decoded_token = auth.verify_id_token(token) # Validate token here
    #     session['user'] = decoded_token # Add user to session
    #     return redirect(url_for('dashboard'))

    try:
         decoded_token = auth.verify_id_token(token)  # Validate token here
         email = decoded_token['email']
         username = email.split('@')[0]  # Extract username from email

         session['user'] = {
            'uid': decoded_token['uid'],  # Store user ID
            'email': email,  # Store user email
            'username': username  # Store extracted username
        }
         return redirect(url_for('dashboard'))

    except Exception as e:
     print(f"Error verifying token: {e}")  # Log the error for debugging
    return "Unauthorized", 401
    
    # except:
    #     return "Unauthorized", 401


#####################
""" Public Routes """

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login')
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html')

@app.route('/signup')
def signup():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    else:
        return render_template('signup.html')


@app.route('/reset-password')
def reset_password():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    else:
        return render_template('forgot_password.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove the user from session
    response = make_response(redirect(url_for('login')))
    response.set_cookie('session', '', expires=0)  # Optionally clear the session cookie
    return response


##############################################
""" Private Routes (Require authorization) """


@app.route('/dashboard')
@auth_required
def dashboard():

    return render_template('dashboard.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/translation')
def get_translation():
    global translation_txt
    return jsonify({'translation': translation_txt})

@app.route('/stop_camera')
def stop_camera():
    global camera
    camera.release()
    camera = None
    print("Camera stopped successfully.")
    return 'Camera Stopped', 200

if __name__ == '__main__':
    app.run(debug=True)