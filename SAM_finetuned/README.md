#  Surface Water Detection in Farm Fields using SAM

This project uses cutting-edge AI to **detect and highlight surface water** in farm fields using satellite or drone images. It's especially helpful for **farmers** to monitor water availability and plan irrigation more efficiently.

---

##  What Does It Do?

When you upload an image of a farm field, this tool automatically **detects the presence of surface water** and displays it using a **clear visual mask**. This helps you easily see where water is present in your field.

---

##  What Makes It Special?

- ✅ Uses **SAM (Segment Anything Model)** from Meta AI, known for accurate image segmentation.
- ✅ Fine-tuned using **100 well-annotated images** of surface water in farm environments.
- ✅ Based on **SAM-ViT-B** (a Vision Transformer model).
- ✅ Provides **pixel-level precision** by creating a mask over the water surface.

---

## 🧠 How Does It Work?

1. You upload an image from your drone, phone, or satellite view.
2. The model analyzes the image and **detects surface water**.
3. It generates a **mask** that highlights all the water regions in the image.
4. You can view the result to guide water management decisions.

---

## 🛠️ What’s Inside the Project?

- `sam_infer.py` – processes input images using the fine-tuned SAM model.
- `surface_water_model.pth` – the fine-tuned model weights.
- `utils/overlay.py` – code to overlay the mask on top of the original image.
- `app.py` – frontend interface for user interaction.
- `requirements.txt` – list of required libraries and tools.
