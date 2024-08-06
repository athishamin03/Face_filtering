import cv2
import cvzone
from PIL import Image
import numpy as np

def main():
    # Initialize the video capture and load the cascade classifier
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load overlay images
    overlay_paths = ['cool.png', 'beard.png', 'sunglass.png','native.png', 'pirate.png']
    overlays = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in overlay_paths]
    current_filter_index = 0
    image_counter = 0
    video_counter = 0
    recording = False

    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = cascade.detectMultiScale(gray_scale, scaleFactor=1.1, minNeighbors=5)

        # Check if any faces are detected
        for (x, y, w, h) in faces:
            # Resize the overlay image to fit the face
            overlay_resize = cv2.resize(overlays[current_filter_index], (int(w * 1.5), int(h * 1.5)))

            # Overlay the image on the face
            frame = cvzone.overlayPNG(frame, overlay_resize, [x - 45, y - 75])

        # Display the frame
        cv2.imshow('Filter', frame)

        # Check for key presses
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        elif key == ord('n'):
            # Switch to the next filter
            current_filter_index = (current_filter_index + 1) % len(overlays)
        elif key == ord('p'):
            # Switch to the previous filter
            current_filter_index = (current_filter_index - 1) % len(overlays)
        elif key == ord('c'):
            # Save the image using Pillow
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            filename = f'captured_image_{image_counter}.png'
            pil_image.save(filename)
            print(f'Image saved as {filename}')
            image_counter += 1
        elif key == ord('r'):
            # Start or stop recording
            if not recording:
                recording = True
                video_filename = f'recorded_video_{video_counter}.avi'
                out = cv2.VideoWriter(video_filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                print(f'Recording started. Saving to {video_filename}')
            else:
                recording = False
                if out is not None:
                    out.release()
                    print('Recording stopped.')
                out = None

        # Write the frame to the video file if recording
        if recording and out is not None:
            out.write(frame)

    # Release the video capture object and close all OpenCV windows
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
