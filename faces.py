import cv2

from cv2 import CASCADE_SCALE_IMAGE

from face_catcher import FaceCatcher, ConfidenceFilter, TimeFilter

cascPath = '/Users/will/miniconda3/envs/faces/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml'


faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
CONFIDENCE_THRESHOLD = 5
catcher = FaceCatcher(filters=[
    ConfidenceFilter(threshold=CONFIDENCE_THRESHOLD),
    TimeFilter(seconds=4)
])

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces, reject_levels, level_weights = \
        faceCascade.detectMultiScale3(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=CASCADE_SCALE_IMAGE,
            outputRejectLevels=True
        )

    # Draw a rectangle around the faces
    # color is in (b, g, r)
    orange = (63, 205, 252)
    green = (64, 255, 60)
    for (x, y, w, h), weight in zip(faces, level_weights):

        # Color based on confidence
        if weight[0] > CONFIDENCE_THRESHOLD:
            color = green
        else:
            color = orange

        # Save the face
        catcher.catch(
            image=frame[y:y+h, x:x+w],
            confidence=weight[0],
        )

        cv2.rectangle(frame, (x, y), (x+w, y+h), color=color, thickness=2)
        cv2.putText(
            img=frame,
            text=f"{weight[0]:0.1f}",
            org=(x,y),
            fontFace=0,
            fontScale=0.8,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
            # bottomLeftOrigin=None,
        )


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()