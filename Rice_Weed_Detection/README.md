# 🌿 Rice Field Weed Detection

This project helps **identify and highlight weeds** in rice fields using advanced AI technology. It is built to assist **farmers** by providing quick and visual detection of unwanted plants (weeds) growing among rice crops.

---

## 🧐 What Does It Do?

By analyzing photos of rice fields, this system can **detect different types of weeds** and **highlight them with colored masks**. This makes it easier to identify where the weeds are and take action accordingly.

---

## 🌾 What Makes It Special?

- ✅ **Trained on 5000 images** of rice fields with multiple types of weeds.
- ✅ Uses **YOLO (You Only Look Once)** — a fast and accurate object detection model.
- ✅ All images were **well-annotated**, meaning the system learned from high-quality, labeled data.
- ✅ The output shows **colored masks** over weeds for clear visual understanding.

---


## 🛠️ What’s Inside the Project?

- `train.py` – trains the YOLO model using annotated images.
- `predict.py` – detects weeds in new images.
- `visualize.py` – applies colored masks over the detected weeds.
- `requirements.txt` – list of tools and libraries needed to run the project.
