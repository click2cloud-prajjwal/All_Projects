# 🐛 Rice Pest Detection

This project helps **identify rice pests** from photos of rice leaves using artificial intelligence (AI). It's designed to assist **farmers and agronomists** in quickly detecting pest infestations in rice fields.

---

##  What Does It Do?

With this tool, you can **upload a photo of a rice leaf or pest**, and it will **identify which pest is affecting the plant**. This helps in making faster and more accurate decisions for pest control.

---

##  Pests It Can Detect:

| **Class**           | **Description**                                                                | **Image Count** |
|---------------------|---------------------------------------------------------------------------------|-----------------|
| **Mealybug**        | Small, white, wax-coated insects that suck sap and cause stunted growth.       | 441             |
| **Thrips**          | Tiny slender insects that feed on plant tissues, causing silvering and curling.| 576             |
| **Rice Leaf Roller**| Larvae that roll and feed inside the leaf, damaging photosynthesis.            | 711             |
| **Hopper**          | Sap-sucking pests causing hopper burn and yellowing of leaves.                 | 1,490           |
| **Borers**          | Stem-boring pests that cause dead hearts and whiteheads.                       | 1,076           |
| **Grasshopper**     | Large chewing insects that feed on leaves and tender shoots.                   | 396             |

**Total images used for training**: **4,691**

---

## 🧠 How Does It Work?

- The system was trained on a **custom dataset of 4,691 images** of rice pest symptoms.
- It uses **EfficientNet-B3**, a powerful deep learning model known for its efficiency and accuracy.
- The model was fine-tuned specifically for rice pest recognition.
- You can **upload an image**, and the system will **predict the pest affecting the plant**.

---

## 🛠️ What’s Included in the Project?

- `Efficientnet-b3-train.py` – trains the pest detection AI model.
- `app.py` – Streamlit app for testing the model locally.
- `class_map.py` – maps prediction indices to actual pest class names.
- `final_api.py` – production-ready API code deployed to Agripilot.
- `train_test_split.py` – script to split the dataset into training and test sets.
