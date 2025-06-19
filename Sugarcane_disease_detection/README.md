# 🌱 Sugarcane Disease Detection

This project helps **identify sugarcane plant diseases** from photos of sugarcane leaves using artificial intelligence (AI). It's built to assist **farmers** in quickly finding and recognizing problems with sugarcane crops.

---

##  What Does It Do?

With this tool, you can **upload a photo of a sugarcane leaf**, and it will **tell you if the plant is healthy or affected by a disease**. If a disease is found, the system will identify which one it is from a list of common sugarcane diseases.

---

##  Diseases It Can Detect:

| **Class**        | **Description**                                                                | **Image Count** |
| ---------------- | ------------------------------------------------------------------------------ | --------------- |
| **Grassy Shoot** | Characterized by excessive tillering and overall stunted growth.               | 567             |
| **Healthy**      | Represents sugarcane plants free from visible disease symptoms.                | 625             |
| **Mosaic**       | Displays irregular patterns of light and dark green areas on the leaves.       | 620             |
| **Pokkah Boeng** | Leads to twisted, malformed leaves with reddish or purple stripes.             | 504             |
| **Red Rot**      | A severe fungal disease marked by red discoloration within the stalk tissue.   | 635             |
| **Rust**         | Appears as small, orange to brown pustules or spots on the leaf surface.       | 625             |
| **Smut**         | Identified by the presence of long, black, whip-like structures on the plant.  | 931             |
| **Yellow Leaf**  | Shows yellowing symptoms, especially along the leaf midrib.                    | 622             |
| **Unknown**      | Used for non-sugarcane crops or instances where the disease status is unclear. | 368             |


## 🧠 How Does It Work?

- The system was trained on **8987 images** of sugarcane leaves.
- It uses **EfficientNet-B3**, an AI model known for its speed and accuracy in understanding images.
- Multiple models are combined using **model ensembling**, which means results from different models are merged to make better predictions.
- You just need to **upload an image**, and the system will **analyze it and tell you the disease**.

---

## 🛠️ What’s Included in the Project?

- `Efficientnetb3-train.py` – trains the main AI model.
- `fine-tune_16-06.py` - new updated training model script with new unknown class
- `model_ensemble.py` – combines predictions from multiple models.
- `app.py` – the interface that users interact with.
- `requirements.txt` – lists tools needed to run the project.
