import os
import cv2
import numpy as np
from PIL import Image

dataset_path = "datasets"
trainer_path = "trainer"
os.makedirs(trainer_path, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()

face_samples = []
ids = []
label_map = {}
current_id = 0

for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue

    label_map[current_id] = person_name

    for file_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, file_name)

        try:
            pil_img = Image.open(img_path).convert("L")
            img_numpy = np.array(pil_img, "uint8")
            img_numpy = cv2.resize(img_numpy, (200, 200))
            face_samples.append(img_numpy)
            ids.append(current_id)
        except Exception as e:
            print(f"跳过文件 {img_path}，原因：{e}")

    current_id += 1

if len(face_samples) == 0:
    print("没有可训练的人脸数据，请先运行 collect_faces.py")
    exit()

recognizer.train(face_samples, np.array(ids))
recognizer.save(os.path.join(trainer_path, "trainer.yml"))

with open(os.path.join(trainer_path, "labels.txt"), "w", encoding="utf-8") as f:
    for k, v in label_map.items():
        f.write(f"{k},{v}\n")

print("训练完成！模型已保存到 trainer/trainer.yml")
print("标签映射已保存到 trainer/labels.txt")