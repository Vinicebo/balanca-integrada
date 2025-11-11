import os
import csv
from ultralytics import YOLO
import cv2

# Caminho do modelo YOLO treinado
MODEL_PATH = "C:/Users/vinic/OneDrive/Documentos/GitHub/balanca-integrada/best.pt"
model = YOLO(MODEL_PATH)

# Caminho do diretório de teste
DATASET_PATH = "C:/Users/vinic/OneDrive/Documentos/GitHub/balanca-integrada/dataset_teste"
OUTPUT_CSV = "resultados_teste.csv"

# Cria ou sobrescreve o CSV
with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["arquivo", "gt_label", "pred_label", "p_top1"])

    # Percorre todas as pastas (cada uma é o ground truth)
    for gt_label in os.listdir(DATASET_PATH):
        pasta = os.path.join(DATASET_PATH, gt_label)
        if not os.path.isdir(pasta):
            continue

        for img_file in os.listdir(pasta):
            if not img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(pasta, img_file)
            results = model(img_path, verbose=False)

            # Extrai predição principal
            detections = results[0].boxes.data.cpu().numpy()
            if len(detections) > 0:
                best = detections[detections[:, 4].argmax()]
                conf = float(best[4])
                class_id = int(best[5])
                pred_label = model.names[class_id]
            else:
                conf = 0.0
                pred_label = "nenhum"

            writer.writerow([img_file, gt_label, pred_label, conf])

print(f"Resultados salvos em {OUTPUT_CSV}")
