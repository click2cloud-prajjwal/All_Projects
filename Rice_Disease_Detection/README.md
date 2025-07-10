# 🌾 Rice Disease Detection

This project helps **identify rice plant diseases** from photos of rice leaves using artificial intelligence (AI). It's designed to help **farmers** quickly detect and recognize problems in rice crops.

---

## 📸 What Does It Do?

With this tool, you can **upload a photo of a rice leaf**, and it will **tell you if the plant is healthy or affected by a disease**. If a disease is found, the system will identify which one it is from a list of common rice diseases.

---

## 🦠 Diseases It Can Detect

| **Class**                | **Description**                                                               | **Image Count** |
| ------------------------ | ----------------------------------------------------------------------------- | --------------- |
| **Bacterial Leaf Blight** | Caused by bacteria, leading to yellowing and drying of leaves from the tips. | 2,581           |
| **Brown Spot**           | Circular brown spots that weaken the plant and reduce yield.                 | 2,563           |
| **Healthy**              | Represents rice plants free from visible disease symptoms.                   | 2,608           |
| **Leaf Blast**           | Shows diamond-shaped lesions with grey centers and brown margins.            | 2,603           |
| **Leaf Scald**           | Characterized by large lesions with wavy edges that dry leaf tips.           | 2,790           |
| **Narrow Brown Spot**    | Narrow brown streaks along the leaf blades.                                  | 2,538           |
| **Rice Hispa**           | Insect pest that causes scraping and white streaks on leaves.                |   780           |
| **Sheath Blight**        | Causes lesions on leaf sheaths near the water line.                          | 2,720           |

**Total Images:** 19,183

---

## 🧠 How Does It Work?

- The system was trained on **19,183 images** of rice leaves.
- It uses **EfficientNet-B3**, an advanced AI model known for high speed and accuracy.
- Multiple models are combined using **model ensembling**, merging results for more reliable predictions.
- You simply **upload an image**, and the system will **analyze it and detect the disease**.

---

## 🛠️ What’s Included in the Project?

- `Efficientnetb3-train.py` – trains the main AI model.
- `model_ensemble.py` – combines results from different models for better accuracy.
- `app.py` – creates a simple user interface for uploading images and viewing results.
- `requirements.txt` – lists all tools and libraries needed to run the project.

---

## ✅ How to Use

1. Clone this repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run `app.py` to start the interface.
4. Upload a rice leaf image and get the prediction!
