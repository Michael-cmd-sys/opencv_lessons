import argparse
import cv2
import numpy as np

# Command line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True, help="path to input image")
parser.add_argument(
    "-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file"
)
parser.add_argument(
    "-m", "--model", required=True, help="path to Caffe pre-trained model"
)
parser.add_argument(
    "-c",
    "--confidence",
    type=float,
    default=0.5,
    help="minimum probability to filter weak detections",
)
args = vars(parser.parse_args())


# Load serialized model from disk
print("[INFO]: Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# Load input image and create image blob for input image
# by a fixed 300x300 pixels and then normalizing it
img = cv2.imread(args["image"])
height, width = img.shape[:2]
blob = cv2.dnn.blobFromImage(
    cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
)

# Pass blob through net and obtain predictions and detections
print("[INFO] Computing object detections...")
net.setInput(blob)
detections = net.forward()


# Loop over detections
for i in range(0, detections.shape[2]):
    # Extract confidence interval associated with predictions
    confidence = detections[0, 0, i, 2]

    # Filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
    if confidence >= args["confidence"]:
        # Compute (x, y) coordinates for the bounding box of the object
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        startX, startY, endX, endY = box.astype("int")

        # Draw bounding box of shape along with associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(
            img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2
        )


def main():
    print("Hello from face-detection!")
    cv2.imshow("Output", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
