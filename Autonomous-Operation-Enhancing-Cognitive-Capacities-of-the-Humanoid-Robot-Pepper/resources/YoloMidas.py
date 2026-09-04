import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import matplotlib.pyplot as plt

# Charger le modèle MiDaS
from hubconf import MiDaS
model = MiDaS(model_type="DPT_Hybrid").to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Charger l'image
image_path = r"C:\Users\hp\Desktop\Master IA2S\Mini Projet\pythoncode\PepperImages\pepper_image.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Prétraiter l'image pour MiDaS
transform = Compose([
    Resize((384, 384)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_image = transform(image).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

# Estimer la carte de profondeur
with torch.no_grad():
    depth = model(input_image)
depth = depth.squeeze().cpu().numpy()
depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Afficher la carte de profondeur
plt.imshow(depth_normalized, cmap="plasma")
plt.axis("off")
plt.show()

# Détection des objets avec YOLOv8
yolo_model = YOLO("yolov8n.pt")
results = yolo_model(image)

# Combiner les résultats de YOLOv8 et MiDaS
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = yolo_model.names[cls_id]

        # Calculer le centre de la boîte
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Estimer la profondeur au centre de la boîte
        depth_value = depth[center_y, center_x]

        # Afficher les coordonnées 3D
        print(f"Objet: {label}, Coordonnées 3D: ({center_x}, {center_y}, {depth_value})")

        # Dessiner la boîte et l'étiquette
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Afficher l'image avec les détections
cv2.imshow("Résultats", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
