# pip install ultralytics torchvision pillow numpy scikit-learn tabulate tqdm
# python3 tests/two-stage(yolo-resnet).py --data ' --yolo_weights --resnet_weights --use_resnet50

import os
import cv2
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet152, resnet50
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score
import time
from collections import defaultdict
from tabulate import tabulate
from tqdm import tqdm
import csv
import requests
import sys

def test_resnet(data_path, yolo_weights, resnet_weights, model="resnet152", species_names=None, output_dir="output"):
    """
    Run the two-stage detection and classification test
    
    Args:
        data_path (str): Path to the test directory
        yolo_weights (str): Path to the YOLO model file
        resnet_weights (str): Path to the ResNet model file
        model (str): Model type, either "resnet50" or "resnet152"
        species_names (list): List of species names
        output_dir (str): Directory to save output CSV files
    """    
    use_resnet50 = model == "resnet50"
    classifier = TestTwoStage(yolo_weights, resnet_weights, use_resnet50=use_resnet50, 
                             species_names=species_names, output_dir=output_dir)
    classifier.run(data_path)

class TestTwoStage:
    def __init__(self, yolo_model_path, resnet_model_path, num_classes=9, use_resnet50=False, species_names="", output_dir="output"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.yolo_model = YOLO(yolo_model_path)
        self.classification_model = resnet50(pretrained=False) if use_resnet50 else resnet152(pretrained=False)
    
        self.classification_model.fc = nn.Sequential(
            nn.Dropout(0.4),  # Using dropout probability of 0.4 as in training
            nn.Linear(self.classification_model.fc.in_features, num_classes)
        )
        
        state_dict = torch.load(resnet_model_path, map_location=self.device)
        self.classification_model.load_state_dict(state_dict)
        self.classification_model.to(self.device)
        self.classification_model.eval()

        self.classification_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.species_names = species_names
        
    def get_frames(self, test_dir):
        image_dir = os.path.join(test_dir, "images")
        label_dir = os.path.join(test_dir, "labels")
        
        predicted_frames = []
        true_frames = []
        image_names = []

        start_time = time.time()  # Start timing

        for image_name in tqdm(os.listdir(image_dir), desc="Processing Images", unit="image"):
            image_names.append(image_name)
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt'))

            frame = cv2.imread(image_path)
            # Suppress print statements from YOLO model
            with torch.no_grad():
                results = self.yolo_model(frame, conf=0.3, iou=0.5, verbose=False)

            detections = results[0].boxes
            predicted_frame = []

            if detections:
                for box in detections:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    x1, y1, x2, y2 = xyxy[:4]
                    width = x2 - x1
                    height = y2 - y1
                    x_center = x1 + width / 2
                    y_center = y1 + height / 2

                    insect_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    insect_crop_rgb = cv2.cvtColor(insect_crop, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(insect_crop_rgb)
                    input_tensor = self.classification_transform(pil_img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.classification_model(input_tensor)

                    # Directly use the model output without any mapping
                    predicted_class_idx = outputs.argmax(dim=1).item()
                    
                    img_height, img_width, _ = frame.shape
                    x_center_norm = x_center / img_width
                    y_center_norm = y_center / img_height
                    width_norm = width / img_width
                    height_norm = height / img_height
                    predicted_frame.append([predicted_class_idx, x_center_norm, y_center_norm, width_norm, height_norm])

            predicted_frames.append(predicted_frame if predicted_frame else [])

            true_frame = []
            if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                with open(label_path, 'r') as f:
                    for line in f:
                        label_line = line.strip().split()
                        true_frame.append([int(label_line[0]), *map(np.float32, label_line[1:])])

            true_frames.append(true_frame if true_frame else [])

        end_time = time.time()  # End timing

        model_type = "resnet50" if isinstance(self.classification_model, type(resnet50())) else "resnet152"
        output_file = os.path.join(self.output_dir, f"results_{model_type}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(output_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Image Name", "True", "Predicted"])
            for image_name, true_frame, predicted_frame in zip(image_names, true_frames, predicted_frames):
                writer.writerow([image_name, true_frame, predicted_frame])
        
        print(f"Results saved to {output_file}")
        return predicted_frames, true_frames, end_time - start_time
    
    def get_taxonomic_info(self, species_list):
        """
        Retrieves taxonomic information for a list of species from GBIF API.
        Creates a hierarchical taxonomy dictionary with order, family, and species relationships.
        """
        taxonomy = {1: [], 2: {}, 3: {}}
        species_to_family = {}
        family_to_order = {}
        
        print(f"Building taxonomy from GBIF for {len(species_list)} species")
        
        print("\nTaxonomy Results:")
        print("-" * 80)
        print(f"{'Species':<30} {'Order':<20} {'Family':<20} {'Status'}")
        print("-" * 80)
        
        for species_name in species_list:
            url = f"https://api.gbif.org/v1/species/match?name={species_name}&verbose=true"
            try:
                response = requests.get(url)
                data = response.json()
                
                if data.get('status') == 'ACCEPTED' or data.get('status') == 'SYNONYM':
                    family = data.get('family')
                    order = data.get('order')
                    
                    if family and order:
                        status = "OK"
                        
                        print(f"{species_name:<30} {order:<20} {family:<20} {status}")
                        
                        species_to_family[species_name] = family
                        family_to_order[family] = order
                        
                        if order not in taxonomy[1]:
                            taxonomy[1].append(order)
                        
                        taxonomy[2][family] = order
                        taxonomy[3][species_name] = family
                    else:
                        error_msg = f"Species '{species_name}' found in GBIF but family and order not found, could be spelling error in species, check GBIF"
                        print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                        print(f"Error: {error_msg}")
                        sys.exit(1)  # Stop the script
                else:
                    error_msg = f"Species '{species_name}' not found in GBIF, could be spelling error, check GBIF"
                    print(f"{species_name:<30} {'Not found':<20} {'Not found':<20} ERROR")
                    print(f"Error: {error_msg}")
                    sys.exit(1)  # Stop the script
                    
            except Exception as e:
                error_msg = f"Error retrieving data for species '{species_name}': {str(e)}"
                print(f"{species_name:<30} {'Error':<20} {'Error':<20} FAILED")
                print(f"Error: {error_msg}")
                sys.exit(1)  # Stop the script
        
        taxonomy[1] = sorted(list(set(taxonomy[1])))
        print("-" * 80)
        
        num_orders = len(taxonomy[1])
        num_families = len(taxonomy[2])
        num_species = len(taxonomy[3])
        
        print("\nOrder indices:")
        for i, order in enumerate(taxonomy[1]):
            print(f"  {i}: {order}")
        
        print("\nFamily indices:")
        for i, family in enumerate(taxonomy[2].keys()):
            print(f"  {i}: {family}")
        
        print("\nSpecies indices:")
        for i, species in enumerate(species_list):
            print(f"  {i}: {species}")
        
        print(f"\nTaxonomy built: {num_orders} orders, {num_families} families, {num_species} species")
        
        return taxonomy, species_to_family, family_to_order
    
    def get_metrics(self, predicted_frames, true_frames, labels):
        """
        Calculate precision, recall, and F1 score for object detection results.
        """
        def calculate_iou(box1, box2):
            x1_min, y1_min = box1[1] - box1[3] / 2, box1[2] - box1[4] / 2
            x1_max, y1_max = box1[1] + box1[3] / 2, box1[2] + box1[4] / 2
            x2_min, y2_min = box2[1] - box2[3] / 2, box2[2] - box2[4] / 2
            x2_max, y2_max = box2[1] + box2[3] / 2, box2[2] + box2[4] / 2

            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)

            iou = inter_area / (box1_area + box2_area - inter_area)
            return iou

        def calculate_precision_recall(pred_boxes, true_boxes, iou_threshold=0.5):
            label_results = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
            generic_tp = 0
            generic_fp = 0
            
            matched_true_boxes = set()
            
            for pred_box in pred_boxes:
                label_idx = pred_box[0]
                matched = False
                
                best_iou = 0
                best_match_idx = -1
                
                for i, true_box in enumerate(true_boxes):
                    if i in matched_true_boxes:
                        continue
                    
                    iou = calculate_iou(pred_box, true_box)
                    if iou >= iou_threshold and iou > best_iou:
                        best_iou = iou
                        best_match_idx = i
                
                if best_match_idx >= 0:
                    matched = True
                    true_box = true_boxes[best_match_idx]
                    matched_true_boxes.add(best_match_idx)
                    generic_tp += 1
                    
                    if pred_box[0] == true_box[0]:
                        label_results[label_idx]['tp'] += 1
                    else:
                        label_results[label_idx]['fp'] += 1
                        true_label_idx = true_box[0]
                        label_results[true_label_idx]['fn'] += 1
                        
                if not matched:
                    label_results[label_idx]['fp'] += 1
                    generic_fp += 1
            
            for i, true_box in enumerate(true_boxes):
                if i not in matched_true_boxes:
                    label_idx = true_box[0]
                    label_results[label_idx]['fn'] += 1
            
            generic_fn = len(true_boxes) - len(matched_true_boxes)
            
            return label_results, generic_tp, generic_fp, generic_fn

        label_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0})
        background_metrics = {'tp': 0, 'fp': 0, 'fn': 0, 'support': 0}
        generic_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
        
        for true_frame in true_frames:
            if not true_frame:  # Empty frame (background only)
                background_metrics['support'] += 1
            else:
                for true_box in true_frame:
                    label_idx = true_box[0]
                    label_metrics[label_idx]['support'] += 1  # Count each detection, not just unique labels

        for pred_frame, true_frame in zip(predicted_frames, true_frames):
            if not pred_frame and not true_frame:
                background_metrics['tp'] += 1
            elif not pred_frame:
                background_metrics['fn'] += 1
            elif not true_frame:
                background_metrics['fp'] += 1
            else:
                frame_results, g_tp, g_fp, g_fn = calculate_precision_recall(pred_frame, true_frame)
                
                for label_idx, metrics in frame_results.items():
                    label_metrics[label_idx]['tp'] += metrics['tp']
                    label_metrics[label_idx]['fp'] += metrics['fp'] 
                    label_metrics[label_idx]['fn'] += metrics['fn']
                
                generic_metrics['tp'] += g_tp
                generic_metrics['fp'] += g_fp
                generic_metrics['fn'] += g_fn

        table_data = []
        # Store individual class metrics for macro-averaging
        class_precisions = []
        class_recalls = []
        class_f1s = []

        for label_idx, metrics in label_metrics.items():
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            support = metrics['support']
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Store for macro-averaging
            class_precisions.append(precision)
            class_recalls.append(recall)
            class_f1s.append(f1_score)
            
            label_name = labels[label_idx] if label_idx < len(labels) else f"Label {label_idx}"
            table_data.append([label_name, f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}", f"{support}"])

            print(f"Debug {label_name}: TP={tp}, FP={fp}, FN={fn}")
            print(f"  Raw P={tp/(tp+fp) if (tp+fp)>0 else 0}, R={tp/(tp+fn) if (tp+fn)>0 else 0}")

        tp = background_metrics['tp']
        fp = background_metrics['fp']
        fn = background_metrics['fn']
        support = background_metrics['support']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        table_data.append(["Background", f"{precision:.2f}", f"{recall:.2f}", f"{f1_score:.2f}", f"{support}"])

        headers = ["Label", "Precision", "Recall", "F1 Score", "Support"]
        total_tp = sum(metrics['tp'] for metrics in label_metrics.values())
        total_fp = sum(metrics['fp'] for metrics in label_metrics.values())
        total_fn = sum(metrics['fn'] for metrics in label_metrics.values())
        total_support = sum(metrics['support'] for metrics in label_metrics.values())

        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0

        table_data.append(["\nTotal (micro-avg, excl. background)", f"{total_precision:.2f}", f"{total_recall:.2f}", f"{total_f1_score:.2f}", f"{total_support}"])
        
        # Add macro-average
        if class_precisions:
            macro_precision = sum(class_precisions) / len(class_precisions)
            macro_recall = sum(class_recalls) / len(class_recalls)
            macro_f1 = sum(class_f1s) / len(class_f1s)
            table_data.append(["Total (macro-avg, excl. background)", f"{macro_precision:.2f}", f"{macro_recall:.2f}", f"{macro_f1:.2f}", f"{total_support}"])
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
        generic_tp = generic_metrics['tp']
        generic_fp = generic_metrics['fp']
        generic_fn = generic_metrics['fn']

        generic_precision = generic_tp / (generic_tp + generic_fp) if (generic_tp + generic_fp) > 0 else 0
        generic_recall = generic_tp / (generic_tp + generic_fn) if (generic_tp + generic_fn) > 0 else 0
        generic_f1_score = 2 * (generic_precision * generic_recall) / (generic_precision + generic_recall) if (generic_precision + generic_recall) > 0 else 0

        print("\nGeneric Total", f"{generic_precision:.2f}", f"{generic_recall:.2f}", f"{generic_f1_score:.2f}")
        
        return total_precision, total_recall, total_f1_score

    def run(self, test_dir):
        predicted_frames, true_frames, total_time = self.get_frames(test_dir)
        num_frames = len(os.listdir(os.path.join(test_dir, 'images')))
        avg_time_per_frame = total_time / num_frames

        print(f"\nTotal time: {total_time:.2f} seconds")
        print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")
        
        # Get taxonomy information for hierarchical analysis
        taxonomy, species_to_family, family_to_order = self.get_taxonomic_info(self.species_names)
        family_list = list(family_to_order.keys())
        order_list = list(taxonomy[1])
        
        # Convert species-level predictions to family and order levels
        true_family_frames = []
        true_order_frames = []
        predicted_family_frames = []
        predicted_order_frames = []
        
        for true_frame in true_frames:
            frame_family_boxes = []
            frame_order_boxes = []
            
            if true_frame:
                for true_box in true_frame:
                    species_idx = true_box[0]
                    species_name = self.species_names[species_idx]
                    family_name = species_to_family[species_name]
                    order_name = family_to_order[family_name]
                    
                    family_label = [family_list.index(family_name)] + list(true_box[1:])
                    order_label = [order_list.index(order_name)] + list(true_box[1:])
                    
                    frame_family_boxes.append(family_label)
                    frame_order_boxes.append(order_label)
                    
            true_family_frames.append(frame_family_boxes)
            true_order_frames.append(frame_order_boxes)
        
        for pred_frame in predicted_frames:
            frame_family_boxes = []
            frame_order_boxes = []
            
            if pred_frame:
                for pred_box in pred_frame:
                    species_idx = pred_box[0]
                    species_name = self.species_names[species_idx]
                    family_name = species_to_family[species_name]
                    order_name = family_to_order[family_name]
                    
                    family_label = [family_list.index(family_name)] + list(map(np.float32, pred_box[1:]))
                    order_label = [order_list.index(order_name)] + list(map(np.float32, pred_box[1:]))
                    
                    frame_family_boxes.append(family_label)
                    frame_order_boxes.append(order_label)
                    
            predicted_family_frames.append(frame_family_boxes)
            predicted_order_frames.append(frame_order_boxes)
        
        # Display metrics for all taxonomic levels
        print("\nSpecies Level Metrics")
        self.get_metrics(predicted_frames, true_frames, self.species_names)
        
        print("\nFamily Level Metrics")
        self.get_metrics(predicted_family_frames, true_family_frames, family_list)
        
        print("\nOrder Level Metrics")
        self.get_metrics(predicted_order_frames, true_order_frames, order_list)

if __name__ == "__main__":
    species_names = [
        "Coccinella septempunctata", "Apis mellifera", "Bombus lapidarius", "Bombus terrestris",
        "Eupeodes corollae", "Episyrphus balteatus", "Aglais urticae", "Vespula vulgaris",
        "Eristalis tenax"
    ]
    
    test_resnet(
        data_path="/mnt/nvme0n1p1/mit/two-stage-detection/bjerge-test",
        yolo_weights="/mnt/nvme0n1p1/mit/two-stage-detection/small-generic.pt",
        resnet_weights="/mnt/nvme0n1p1/mit/two-stage-detection/output/best_resnet50.pt",
        model="resnet50",
        species_names=species_names,
        output_dir="/mnt/nvme0n1p1/mit/two-stage-detection/output"
    )