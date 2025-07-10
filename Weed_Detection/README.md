# 🌿 Rice & Sugarcane Weed Detection

This project helps **identify and highlight weeds** in **rice and sugarcane fields** using advanced AI technology. It is designed to assist **farmers** by providing quick, accurate, and visual detection of unwanted plants (weeds) growing among crops.

---

## 📸 What Does It Do?

By analyzing photos of rice or sugarcane fields, this system can **detect different types of weeds** and **highlight them with colored masks**. This makes it easy for farmers and agronomists to see exactly where weeds are present and manage them efficiently.

---

## ✨ What Makes It Special?

- ✅ **Trained on 10,319 images** of rice fields and **9,355 images** of sugarcane fields with various types of weeds.
- ✅ Uses **YOLO (You Only Look Once)** — a fast and accurate state-of-the-art object detection model.
- ✅ All images were **carefully annotated**, so the model learns from high-quality, precisely labeled data.
- ✅ The output shows **colored masks** over detected weeds for clear visual understanding.

---

## 🛠️ What’s Inside the Project?

- `rice_weed_train.py` – trains the YOLO model for weed detection in rice fields.
- `sugarcane_weed_train_script.py` – trains the YOLO model for weed detection in sugarcane fields.
- `predict.py` – runs the trained model to detect weeds in new field images.
- `visualize.py` – applies colored masks to highlight weeds in the detected areas.
- `requirements.txt` – lists all tools and libraries needed to run the project.

---

## ✅ How to Use

1. Clone this repository.
2. Install the dependencies:  
   ```bash
   pip install -r requirements.txt
