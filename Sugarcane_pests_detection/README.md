# 🐛 Sugarcane Pest Detection

This project helps **identify sugarcane pests** from photos of sugarcane leaves using artificial intelligence (AI). It's designed to assist **farmers and agronomists** in quickly detecting pest infestations in sugarcane fields.

---

##  What Does It Do?

With this tool, you can **upload a photo of a sugarcane leaf or pest**, and it will **identify which pest is affecting the plant**. This helps in making faster and more accurate decisions for pest control.

---

##  Pests It Can Detect:

| **Class**       | **Description**                                                                 | **Image Count** |
|-----------------|----------------------------------------------------------------------------------|-----------------|
| **Borer**       | Insects that bore into the stem, causing dead hearts and yield loss.            | 670             |
| **Leaf Mite**   | Tiny arachnids that feed on the undersides of leaves, causing discoloration.    | 552             |
| **Whiteflies**  | Small white-winged insects that suck sap, often leading to sooty mold.          | 771             |
| **Wooly Aphids**| Covered in waxy wool-like material, they cluster on stems and feed on sap.      | 599             |
| **Pyrilla**     | A leaf hopper that causes hopper burn due to sap sucking.                       | 571             |
| **White Grub**  | Soil-dwelling larvae that feed on roots, reducing plant vigor.                  | 525             |

**Total images used for training**: **3,688**

---

## 🧠 How Does It Work?

- The system was trained on a **custom dataset of 3,688 images** of sugarcane pest symptoms.
- It uses **EfficientNet-B3**, a powerful deep learning model known for its efficiency and accuracy.
- The model was fine-tuned specifically for sugarcane pest recognition.
- You can **upload an image**, and the system will **predict the pest affecting the plant**.

---

## 🛠️ What’s Included in the Project?

- `Efficientnet-b3-train.py` – trains the pest detection AI model.
- `app.py` – Streamlit app for testing the model locally.
- `class_map.py` – maps prediction indices to actual pest class names.
- `final_api.py` – production-ready API code deployed to Agripilot.
- `train_test_split.py` – script to split the dataset into training and test sets.

