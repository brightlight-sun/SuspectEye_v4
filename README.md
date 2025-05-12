# SuspectEye_v4
A real time suspect identification through cctv intelligence

| Folder/File                                        | Type                                | Description                                                          |
| -------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------------- |
| `SuspectEyeApp/`                                   | Folder (Root)                       | Main project directory                                               |
| ├── `suspecteye.exe`                               | Executable                          | Final bundled `.exe` file (output from auto-py-to-exe)               |
| ├── `app.py`                                       | Python Script                       | Flask app script to serve the web interface                          |
| ├── `add_faces_v4.py`                              | Python Script                       | Script to add new faces to the system  using device camera           |
| ├── `detect_faces_v4.py`                           | Python Script                       | Main face recognition and alert script                               |
| ├── `add_photo_face.py`                            | Python Script                       | script to add new face to the system using jpg image input           |
| ├── `beepSound.wav`                                | Audio File                          | Beep sound for detection alerts                                      |
| ├── `requirements.txt` (optional)                  | Config File                         | For listing Python packages, if needed for debugging or reinstalling |
| ├── `models/`                                      | Folder                              | Contains face detection models                                       |
| │   ├── `deploy.prototxt.txt`                      | Model Config                        | Caffe model config file                                              |
| │   ├── `res10_300x300_ssd_iter_140000.caffemodel` | Pre-trained Caffe model             |                                                                      |  
| ├── `data/`                                        | Folder                              | Stores serialized encodings and names                                |
| │   ├── `face_encodings.pkl`                       | Pickle File                         | Encoded face data                                                    |
| │   └── `names.pkl`                                | Pickle File                         | Names corresponding to encodings                                     |
| ├── `temp/`                                        | Folder                              | Temporary images saved before Telegram upload                        |
| ├── `templates/`                                   | Folder                              | HTML templates for Flask                                             |
| │   └── `index.html`                               | HTML File                           | Web interface                                                        |
| └── `static/`                                      | Folder (Optional)                   | For CSS, JS, images if used in HTML                                  |
| ├── `photos/`                                      | Folder                              | jpg images to add faces using add_photo_face                         | 
-------------------------------------------------------------------------------------------------------------------------------------------------------------------
