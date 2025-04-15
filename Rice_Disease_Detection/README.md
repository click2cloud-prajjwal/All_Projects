# 🌾 Rice Disease Detection

This project helps **identify rice plant diseases** from pictures of rice leaves using artificial intelligence (AI). It's designed to support **farmers** in detecting crop health problems quickly and accurately.

---

## 🧐 What Does It Do?

By uploading a photo of a rice leaf, this system can **automatically tell if the plant is healthy or infected** — and if it is infected, it can **identify the type of disease** from 8 possible types.

---

## 🦠 Diseases It Can Detect:

1. **Bacterial Leaf Blight** – causes yellowing and drying of leaves  
2. **Brown Spot** – shows up as small brown patches  
3. **Leaf Blast** – creates diamond-shaped grey spots  
4. **Leaf Scald** – long stripes that look burned  
5. **Narrow Brown Spot** – thin, brown lines or spots  
6. **Neck Blast** – affects the rice panicle, drying it out  
7. **Sheath Blight** – white or grey lesions on the leaf sheath  
8. **Tungro** – leads to stunted growth and yellow-orange coloring  
9. **Healthy** – no disease found, your plant is doing great!

---

## 🧠 How Does It Work?

- The system is trained using **5000 images** of diseased and healthy rice leaves.
- It uses a powerful AI model called **EfficientNet-B3** that is good at recognizing images.
- We also combine multiple models together in a technique called **model ensembling** to make sure the predictions are more accurate.
- There's a simple interface where you just **upload an image** and the system will **predict the disease** within seconds.

---

## 🛠️ What's Inside This Project?

- `Efficientnetb3-train.py` – the program that trains the AI model.
- `model_ensemble.py` – combines results from different models for better accuracy.
- `app.py` – makes everything work together and creates a simple user interface.
- `requirements.txt` – lists the tools used to run the project.
