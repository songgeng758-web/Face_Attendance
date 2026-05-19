import cv2
import os
import numpy as np

person_name = input("请输入要录入的人名（建议英文，如 songgeng / zhangsan）: ").strip()
save_count_input = input("请输入要采集的图片数量（直接回车默认 50）: ").strip()

if not person_name:
    print("人名不能为空，程序结束")
    exit()

save_count = int(save_count_input) if save_count_input.isdigit() else 50

dataset_dir = os.path.join("datasets", person_name)
os.makedirs(dataset_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("摄像头打开失败，请检查摄像头是否被占用")
    exit()

count = 0
print("按 q 可提前退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(120, 120)
    )

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        file_name = f"face_{count + 1}.jpg"
        img_path = os.path.join(dataset_dir, file_name)

        success, encoded_img = cv2.imencode(".jpg", face_img)

        if success:
            encoded_img.tofile(img_path)
            if os.path.exists(img_path):
                count += 1
                print(f"已保存：{img_path}")
            else:
                print(f"保存失败：{img_path}")
        else:
            print(f"编码失败：{img_path}")

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Collecting: {count}/{save_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        break

    cv2.imshow("Collect Faces", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):
        break

    if count >= save_count:
        break

cap.release()
cv2.destroyAllWindows()
print(f"采集完成，实际成功保存 {count} 张图片，路径：{dataset_dir}")