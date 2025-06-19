# 🌱 Sugarcane Disease Detection

This project helps **identify sugarcane plant diseases** from photos of sugarcane leaves using artificial intelligence (AI). It's built to assist **farmers** in quickly finding and recognizing problems with sugarcane crops.

---

##  What Does It Do?

With this tool, you can **upload a photo of a sugarcane leaf**, and it will **tell you if the plant is healthy or affected by a disease**. If a disease is found, the system will identify which one it is from a list of common sugarcane diseases.

---

##  Diseases It Can Detect:

1. **Grassy Shoot** – causes excessive tillering and stunted growth.     (567 images)  
2. **Healthy** – your sugarcane plant is free from disease.              (625 images)
3. **Mosaic** – shows patterns of light and dark green areas on leaves.  (620 images)
4. **Pokkah Boeng** – leads to twisted leaves and reddish stripes.       (504 images)
5. **Red Rot** – a serious fungal disease causing red patches inside stalks.   (635 images)
6. **Rust** – shows as small orange or brown spots on the leaves.        (625 images)
7. **Smut** – forms black whip-like structures on the plant.             (931 images)
8. **Yellow Leaf** – yellowing of leaves, especially the midrib area.    (622 images)
9. **Unknown** – Tells the crops are not sugarcane or is unknown.        (368 images)
---

## 🧠 How Does It Work?

- The system was trained on **8,000+ images** of sugarcane leaves.
- It uses **EfficientNet-B3**, an AI model known for its speed and accuracy in understanding images.
- Multiple models are combined using **model ensembling**, which means results from different models are merged to make better predictions.
- You just need to **upload an image**, and the system will **analyze it and tell you the disease**.

---

## 🛠️ What’s Included in the Project?

- `Efficientnetb3-train.py` – trains the main AI model.
- `model_ensemble.py` – combines predictions from multiple models.
- `app.py` – the interface that users interact with.
- `requirements.txt` – lists tools needed to run the project.
