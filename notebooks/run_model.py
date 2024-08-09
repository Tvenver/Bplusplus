from ultralytics import YOLO

# Load a model
path_to_model = "model/train2/weights/best.pt"
model = YOLO(path_to_model)  # load a custom model

# Predict with the model
path_to_image = "nabis_test.jpg"

results1 = model(path_to_image, save_txt = False)  # predict on an image

print(results)