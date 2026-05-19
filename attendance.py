import cv2
import os
import csv
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

trainer_file = os.path.join("trainer", "trainer.yml")
labels_file = os.path.join("trainer", "labels.txt")
log_dir = "attendance_logs"

os.makedirs(log_dir, exist_ok=True)

if not os.path.exists(trainer_file):
    print("未找到 trainer.yml，请先运行 train_model.py")
    exit()

if not os.path.exists(labels_file):
    print("未找到 labels.txt，请先运行 train_model.py")
    exit()

# 读取标签映射（训练生成的原始标签，一般是英文文件夹名）
label_map = {}
with open(labels_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            idx, name = line.split(",", 1)
            label_map[int(idx)] = name

# 显示名称映射：英文标签 -> 中文名字
display_name_map = {
    "songgeng": "宋庚"
    # 后续新增人员时继续加
    # "zhangsan": "张三",
    # "lisi": "李四",
}

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_file)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("正在尝试打开摄像头...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("摄像头对象已创建，正在检查是否打开成功...")

if not cap.isOpened():
    print("摄像头打开失败，请检查摄像头是否被其他程序占用")
    exit()

print("摄像头打开成功")

today = datetime.now().strftime("%Y-%m-%d")
log_path = os.path.join(log_dir, f"{today}.csv")

# 如果今天的签到文件不存在，就先创建并写入表头
if not os.path.exists(log_path):
    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["姓名", "签到时间"])

checked_in_today = set()

# 读取今天已有签到记录，避免重复签到
with open(log_path, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if len(row) >= 1:
            checked_in_today.add(row[0])


def draw_chinese_text(img, text, position, text_color=(0, 255, 0), text_size=30):
    """在 OpenCV 图像上绘制中文"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_path = "C:/Windows/Fonts/msyh.ttc"  # 微软雅黑
    font = ImageFont.truetype(font_path, text_size, encoding="utf-8")

    # OpenCV 是 BGR，PIL 是 RGB，需要转换颜色顺序
    rgb_color = (text_color[2], text_color[1], text_color[0])

    draw.text(position, text, font=font, fill=rgb_color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_status_box(img, text, box_color=(0, 255, 0), text_color=(255, 255, 255)):
    """在画面左上角绘制状态提示框"""
    x1, y1, x2, y2 = 20, 20, 360, 80
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, -1)
    img = draw_chinese_text(img, text, (35, 32), text_color=text_color, text_size=28)
    return img


print("按 q 退出程序，也可以直接点击窗口右上角 X 关闭")

last_status_text = "等待识别"
last_status_color = (128, 128, 128)  # 灰色

window_name = "Face Attendance System"

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 收紧检测参数，减少误检
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(150, 150)
    )

    current_detected = False

    for (x, y, w, h) in faces:
        current_detected = True

        face_img = gray[y:y + h, x:x + w]

        # 统一缩放，提升训练图与识别图的一致性
        face_img = cv2.resize(face_img, (200, 200))

        label, confidence = recognizer.predict(face_img)

        # confidence 越小越像
        if confidence < 70:
            raw_name = label_map.get(label, "unknown")
            name = display_name_map.get(raw_name, raw_name)
            color = (0, 255, 0)
            text = name

            if name not in checked_in_today:
                now_time = datetime.now().strftime("%H:%M:%S")
                with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, now_time])

                checked_in_today.add(name)
                print(f"{name} 已签到：{now_time}")
                last_status_text = f"{name} 签到成功"
                last_status_color = (0, 180, 0)
            else:
                last_status_text = f"{name} 今日已签到"
                last_status_color = (0, 140, 255)
        else:
            name = "未知人员"
            color = (0, 0, 255)
            text = name
            last_status_text = "未知人员，禁止签到"
            last_status_color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # 在人脸框上方绘制姓名
        text_y = y - 35 if y - 35 > 0 else y + h + 5
        frame = draw_chinese_text(
            frame,
            text,
            (x, text_y),
            text_color=color,
            text_size=30
        )

    if not current_detected:
        last_status_text = "未检测到人脸"
        last_status_color = (128, 128, 128)

    # 绘制左上角状态框
    frame = draw_status_box(
        frame,
        last_status_text,
        box_color=last_status_color,
        text_color=(255, 255, 255)
    )

    cv2.imshow(window_name, frame)

    # 先判断按键退出
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

    # 再判断窗口是否被用户点击 X 关闭
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
    except cv2.error:
        # 窗口已经被关闭时，这里有时会抛异常，直接退出即可
        break

cap.release()
cv2.destroyAllWindows()
print(f"今日打卡记录已保存在：{log_path}")