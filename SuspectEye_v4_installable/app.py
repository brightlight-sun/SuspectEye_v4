#python app to integrate html file and python files
from flask import Flask, render_template, request
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/add_face', methods=['POST'])
def add_face():
    name = request.form.get("name")
    if name:
        try:
            subprocess.call(['python', 'add_faces_v4.py', name])  # This will call the face addition script
            return render_template("index.html", message=f"Face captured for {name}")
        except Exception as e:
            return render_template("index.html", message=f"Error: {str(e)}")
    return render_template("index.html", message="Name is required")

@app.route('/recognize', methods=['GET'])
def recognize():
    try:
        subprocess.call(['python', 'detect_faces_v4.py'])  # This will call the face recognition script
        return render_template("index.html", message="Recognition completed")
    except Exception as e:
        return render_template("index.html", message=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
