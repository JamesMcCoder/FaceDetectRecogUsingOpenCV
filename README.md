
# FaceDetectRecogUsingOpenCV
Python3 Face Detection and Recognition Using OpenCV

Inspiration from the many coders out there sharing their work and passion.

Aryal Bibek - https://github.com/Aryal007/opencv_face_recognition

This project is running on a Rasperry Pi 3 (Jessie).

I am just starting the core functionality and not clean proper code for now.

I will enhance it to use a USB Camera, but first phase is to have it identify people in existing picture set.
The pictures are just a mass download from Google Pictures.

The conversion from ID to Person Name is just a quick hack but will refactor that out.

This code can run either Eigen, Fisher or LBPH Face Regognizers.

For this reason, there is an extra script to pull all faces, resize and save in greyscale of your test set.  This essentially creates a new test set which you will then train against.

The renameFilesBatch.py input parameters will look like:

rename(dir, pattern, titlePattern)

Example:  rename(r'In', r'*.*', r'AngelinaJolie.1.')
