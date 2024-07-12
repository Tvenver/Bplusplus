from ultralytics import YOLO

# Load a model
path_to_model = "path_to_best.pt model"
model = YOLO(path_to_model)  # load a custom model

# Predict with the model
path_to_image = "path to folder, image, video"
results = model(path_to_image, save_txt = True)  # predict on an image


