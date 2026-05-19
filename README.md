# 人脸识别考勤系统（Face Attendance System）

一个基于 Python 的本地人脸识别考勤系统。通过摄像头采集人脸、训练识别模型，并在签到时自动比对人脸、记录考勤时间，适用于教室点名、小型团队打卡等场景。

## ✨ 功能特性

- **人脸采集**：调用摄像头采集指定人员的人脸图像，自动保存到数据集
- **模型训练**：对采集到的人脸图像进行编码，生成可复用的识别模型
- **实时考勤**：摄像头实时检测并识别人脸，自动写入考勤记录
- **中文显示**：使用微软雅黑字体在画面中正确渲染中文姓名
- **考勤记录**：考勤结果按日期保存，便于后续查询与统计

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| 编程语言 | Python 3 |
| 人脸识别 | face_recognition、dlib |
| 图像处理 | OpenCV、Pillow |
| 数值计算 | NumPy |

## 📁 项目结构

```
Face_Attendance/
├── collect_faces.py      # 人脸采集脚本
├── train_model.py        # 模型训练脚本
├── attendance.py         # 实时考勤主程序
├── datasets/             # 人脸图像数据集（运行后生成，不纳入版本管理）
├── trainer/              # 训练好的模型文件（运行后生成）
├── attendance_logs/      # 考勤记录（运行后生成）
└── requirements.txt      # 项目依赖清单
```

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/你的用户名/Face_Attendance.git
cd Face_Attendance
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **提示**：`face_recognition` 依赖 `dlib`。在 Windows 上若 `dlib` 安装失败，需先安装 Visual Studio C++ 构建工具，或使用预编译的 whl 包安装。

### 3. 运行流程

```bash
# 第一步：采集人脸（按提示输入姓名）
python collect_faces.py

# 第二步：训练识别模型
python train_model.py

# 第三步：启动考勤
python attendance.py
```

## 📝 使用说明

1. 运行 `collect_faces.py`，对着摄像头采集人脸，系统会保存多张图像到 `datasets/`
2. 运行 `train_model.py`，对数据集中的人脸进行编码并生成模型
3. 运行 `attendance.py`，摄像头识别到已录入的人脸后自动签到，记录写入 `attendance_logs/`

## ⚠️ 注意事项

- 采集人脸时请保证光线充足、面部无遮挡，多角度采集可提升识别准确率
- 数据集（人脸图像）涉及个人隐私，已通过 `.gitignore` 排除，不会上传到仓库
- 中文姓名渲染依赖系统字体文件 `msyh.ttc`，请确保该字体可用

## 📌 后续优化方向

- [ ] 增加图形界面（GUI），降低使用门槛
- [ ] 支持考勤数据导出为 Excel 报表
- [ ] 增加活体检测，防止照片冒充
- [ ] 接入数据库，支持多人员管理与历史查询

---

> 本项目为个人学习与实践作品，欢迎交流与建议。
