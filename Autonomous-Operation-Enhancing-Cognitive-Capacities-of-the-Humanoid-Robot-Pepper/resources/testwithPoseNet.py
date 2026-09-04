import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.model import create_model, load_model
from utils.image import get_affine_transform
from detector.apis import BaseDetector

# Charger le modèle CenterPose (pré-entraîné)
MODEL_PATH = r'C:\Users\hp\CenterPose\models\centerpose_resnet_50.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = create_model('res_50', heads={'hm': 80, 'wh': 2, 'reg': 2, 'dep': 1, 'dim': 3, 'rot': 3}, head_conv=256)
model = load_model(model, MODEL_PATH)
model = model.to(device)
model.eval()

# Charger l'image
image_path = r"C:\Users\hp\Desktop\Master IA2S\Mini Projet\pythoncode\PepperImages\pepper_image.png"  # Remplace par ton image
image = cv2.imread(image_path)
orig_img = image.copy()

# Prétraitement de l'image
img_h, img_w = image.shape[:2]
input_h, input_w = 512, 512  # Taille du modèle
c = np.array([img_w / 2., img_h / 2.], dtype=np.float32)
s = max(img_h, img_w) * 1.0
trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
inp_image = cv2.warpAffine(image, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
inp_image = inp_image.astype(np.float32) / 255.0
inp_image = inp_image.transpose(2, 0, 1)[None, :, :, :]
inp_image = torch.from_numpy(inp_image).to(device)

# Prédiction du modèle
with torch.no_grad():
    outputs = model(inp_image)[-1]

# Extraction des résultats (post-processing)
detector = BaseDetector(model)
results = detector.process(outputs)

# Affichage des résultats
for obj in results:
    cls = obj['class']  # Classe de l'objet détecté
    bbox = obj['bbox']  # Coordonnées 2D
    position_3D = obj['position_3D']  # Coordonnées 3D (X, Y, Z)
    rotation_3D = obj['rotation_3D']  # Rotation en degrés (θ, φ, ψ)
    dimensions_3D = obj['dimensions_3D']  # (h, w, d)

    print(f"Objet détecté: {cls}")
    print(f"Coordonnées 3D : {position_3D}")
    print(f"Rotation 3D : {rotation_3D}")
    print(f"Dimensions 3D : {dimensions_3D}\n")

    # Dessiner le bounding box et l'étiquette
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(orig_img, cls, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Affichage de l'image avec détection
plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
