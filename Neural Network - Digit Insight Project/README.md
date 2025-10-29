# Digit Insight Studio

Interactive playground for handwriting intelligence. Digit Insight Studio pairs a fast CNN classifier, a generative VAE, Grad-CAM explainability, and adaptive segmentation so you can sketch any multi-digit number and instantly explore model predictions, saliency maps, and creative variations.


---
## ✨ Highlights
- **Multi-digit understanding** – segment arbitrary number strings and read each digit independently.
- **Rapid training** – train the classifier (≈5 min) and VAE (≈10 min) on CPU thanks to compact architectures and MNIST scale.
- **Explainability baked in** – heatmaps generated via Grad-CAM show the strokes that drive each decision.
- **Creative companion** – a lightweight VAE imagines stylistic alternatives conditioned on your written digits.
- **Polished UI** – Gradio sketchpad front-end with prediction summaries, galleries, and one-click updates.

---
## 🧱 Tech Stack
| Layer | Tools |
| ----- | ----- |
| Core ML | PyTorch, Torchvision |
| Data | MNIST (handwritten digits) |
| Explainability | Grad-CAM |
| Generative | Convolutional Variational Autoencoder |
| Demo UI | Gradio 5 |
| Utilities | NumPy, SciPy, Matplotlib |

---
## 🚀 Quickstart
```bash
# 1. Clone & enter the project
git clone https://github.com/<your-username>/digit-insight-studio.git
cd digit_insight_studio

# 2. Create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

> First run of any script will download MNIST automatically (~11 MB).

---
## 🧠 Train the Classifier
```bash
python train_classifier.py --config config_classifier.yaml
```
- Architecture: 3-stage CNN with global pooling and dropout
- Default: 5 epochs, AdamW optimizer, data augmentation (rotation + translate)
- Outputs: `checkpoints/classifier.pth` and metrics under `results/`

Evaluate on the test split:
```bash
python evaluate_classifier.py --config config_classifier.yaml --checkpoint checkpoints/classifier.pth
```
Generates `results/test_predictions.png` with a sample grid.

---
## 🎨 Train the VAE
```bash
python train_vae.py --config config_vae.yaml
```
- Convolutional encoder/decoder, latent dim = 16 (configurable)
- Saves best checkpoint to `checkpoints/vae.pth`
- Periodically writes sample grids to `results/vae_samples/`

---
## 🕹️ Launch the Interactive Studio
```bash
python demo.py --classifier checkpoints/classifier.pth --vae checkpoints/vae.pth
```
What you get:
1. Draw any number in the sketchpad (e.g., **2025** or **314159**)
2. Instant prediction string with per-digit confidence
3. Grad-CAM overlays per digit (gallery)
4. VAE-generated variations for each digit (gallery)

---
## 🧩 How it Works
1. **Segmentation** (`src/utils/segmentation.py`)
   - Threshold + morphological cleanup
   - Connected-component analysis
   - Aspect-preserving cropping with adaptive padding
2. **Classification** (`src/models/classifier.py`)
   - CNN extracts spatial features
   - Global average pooling → dense head → 10-way logits
3. **Explainability** (`src/utils/gradcam.py`)
   - Grad-CAM heatmap computed per digit tensor
   - Visual overlay rendered via Matplotlib (`src/utils/visualization.py`)
4. **Generation** (`src/models/vae.py`)
   - Encodes each digit → latent space → decodes multiple samples
5. **Frontend** (`demo.py`)
   - Gradio Sketchpad input → segmentation pipeline
   - Aggregates results into text summary and galleries

---
## 📁 Repository Layout
```
digit_insight_studio/
├── README.md
├── requirements.txt
├── config_classifier.yaml
├── config_vae.yaml
├── train_classifier.py
├── train_vae.py
├── evaluate_classifier.py
├── demo.py
├── checkpoints/              # saved model weights (gitignored)
├── results/                  # metrics, grids, demo assets (gitignored)
└── src/
    ├── __init__.py
    ├── data.py               # MNIST dataloaders + transforms
    ├── models/
    │   ├── __init__.py
    │   ├── classifier.py
    │   └── vae.py
    └── utils/
        ├── __init__.py
        ├── gradcam.py
        ├── segmentation.py
        └── visualization.py
```

---
## ⚙️ Configuration
- **`config_classifier.yaml`**
  - `data.batch_size`, `data.augment`, `training.epochs`, `model.hidden_channels`
- **`config_vae.yaml`**
  - `model.latent_dim`, `training.beta`, `training.epochs`
- Segmentation tweaks in `SegmentConfig` (threshold, min_pixels, padding) if you want to adapt to other handwriting styles.

---
## 🛣️ Roadmap Ideas
- Support EMNIST for letters → turn digits into alphanumeric OCR
- Bundle Grad-CAM + VAE outputs into downloadable reports
- Deploy the demo on Hugging Face Spaces for easy sharing
- Add a CRNN backend to output digits end-to-end without explicit segmentation

---
## 📜 License
MIT License © 2025 Adityaa Bandaru

---
## 🙌 Acknowledgments
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
- [Grad-CAM](https://arxiv.org/abs/1610.02391)
- Inspiration from classic OCR pipelines and modern explainable AI tooling.
