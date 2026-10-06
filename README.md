# 🔎 OCR with Qwen2-VL on AMD ROCm

A vision-language OCR solution developed for the **Lablab x AMD AI Academy Challenge – Mini Challenge 2**.

The project uses **Qwen2-VL-7B-Instruct** to recognize text from real-world images, including license plates and road signs, while running inference on AMD GPUs through ROCm.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| 🐍 Python | Application development |
| 🤗 Hugging Face Transformers | Model loading and inference |
| 👁️ Qwen2-VL-7B-Instruct | Vision-language OCR model |
| 🔥 PyTorch | Deep learning framework |
| ⚡ AMD ROCm | GPU acceleration |
| 🐳 Docker | Containerization and deployment |
| 🚀 AMD Instinct MI300X | GPU used for testing |

---

## ✨ Features

- 📝 OCR for license plates and road signs
- 🇺🇸 US license plate recognition
- 🇨🇳 Chinese license plate recognition
- 🚦 Traffic sign text recognition
- 🔢 Recognition of numbers and advisory speed plaques
- 🌙 Handles low-light images
- 🌫️ Handles blur and noisy images
- 💡 Handles glare
- 📐 Supports off-axis views
- 📖 Preserves reading order
- 🎯 Extracts only the relevant target text
- 🐳 Ready-to-run Docker container for evaluation

---

## 🧠 Model

**Qwen2-VL-7B-Instruct**

The project uses the pretrained instruction-tuned Qwen2-VL model with an OCR-specific prompt to guide the model toward extracting only the visible text belonging to the target object.

> This project uses prompt engineering rather than fine-tuning.

### Model Configuration

- Model: `Qwen/Qwen2-VL-7B-Instruct`
- Precision: FP16
- Framework: PyTorch
- Inference: Hugging Face Transformers
- Device: AMD GPU
- GPU Runtime: ROCm 10.0

---

## 🐳 Docker

The application is packaged using the **AMD-mandated ROCm PyTorch base image**:

```dockerfile
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
```

### Public Docker Image

```text
mohanad06/ocr-qwen2vl:latest
```

The image is publicly available on Docker Hub and can be pulled using:

```bash
docker pull mohanad06/ocr-qwen2vl:latest
```

---

## 🚀 Usage

The container accepts an input image through the required command-line interface:

```bash
python3 /app/app.py --input-image /app/input/image_01.png
```

The OCR result is written to:

```text
/app/output/image_01_output.json
```

### Example Output

```json
{
  "text": "7ABC123"
}
```

---

## 📂 Project Structure

```text
ocr-qwen2vl/
│
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

### File Description

| File | Description |
|---|---|
| `app.py` | OCR inference application |
| `Dockerfile` | AMD ROCm container configuration |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |

---

## 🧪 Testing

The solution was tested using the provided OCR challenge examples, including:

- US license plates
- Chinese license plates
- Stop signs
- Speed limit signs
- Work zone signs
- Advisory speed plaques
- Blurred images
- Low-light images
- Glare and noisy images

The Docker container was successfully tested on an **AMD Instinct MI300X** GPU with **ROCm 10.0**.

---

## ⚡ Performance

During testing on an AMD Instinct MI300X:

- Model loading: approximately **18.7 seconds**
- OCR inference: approximately **0.34 seconds/image**
- Peak VRAM usage: approximately **15.6 GB**

These results were measured during local testing and may vary depending on the evaluation environment and image.

---

## 🏆 Challenge

Developed for:

**Lablab x AMD AI Academy Challenge — Mini Challenge 2: OCR**

The solution follows the challenge requirements for the AMD ROCm base image, Docker packaging, GPU inference, and JSON output format.
