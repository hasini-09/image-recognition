import cv2

# Path to the image
image_path = r"C:/Users/velpu/OneDrive/Desktop/HASINI/lab_projects/python/example_image.jpg"

# Load the image
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Image not found at {image_path}. Please check the path and try again.")
else:
    print("Image loaded successfully!")

    # Load DNN model
    model_path = r"C:/Users/velpu/OneDrive/Desktop/HASINI/python_models/deploy.prototxt"
    weights_path = r"C:/Users/velpu/OneDrive/Desktop/HASINI/python_models/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(model_path, weights_path)

    # Prepare input for the DNN
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

    # Set the input and perform inference
    net.setInput(blob)
    detections = net.forward()

    # Draw detections on the image
    count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            count += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x, y, x1, y1 = box.astype("int")
            cv2.rectangle(image, (x, y), (x1, y1), (0, 255, 0), 2)  # Green rectangle

    print(f"Number of faces detected: {count}")

    # Display the image
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
