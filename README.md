# NeuroScan AI — Brain Tumor Classification

NeuroScan AI is a machine-learning project that classifies brain MRI images into four categories using a trained EfficientNetB0 model.

## Classes

- Glioma
- Meningioma
- No Tumor
- Pituitary Tumor

## Technology Stack

**Machine Learning:** TensorFlow, Keras, EfficientNetB0  
**Backend:** Python, FastAPI  
**Frontend:** React, Vite  
**Image Processing:** Pillow  
**Data Storage:** SQLite for local prediction history

## Project Structure

```text
BrainTumourDetection/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── docs/
│   └── evaluation/
├── frontend/
├── brain_tumor_efficientnetb0.h5
├── class_names_efficientnet.json
├── train_efficientnet.py
├── evaluate_efficientnet.py
├── evaluate_model.py
└── README.md
```

Dataset folders, virtual environments, and local history databases are excluded from version control.

## Model Evaluation

The current EfficientNetB0 model was evaluated on the cleaned test dataset.

| Metric | Result |
|---|---:|
| Test accuracy | 89.58% |
| Macro F1-score | 89.21% |
| Weighted F1-score | 89.28% |
| Test images | 1,584 |

These results describe performance on the evaluated dataset. They do not establish clinical reliability or suitability for medical diagnosis.

### Evaluation Visualizations

Evaluation plots from different model experiments are stored in [`docs/evaluation/`](docs/evaluation/). Check each plot's filename before interpreting it as a result from the current EfficientNetB0 model.

## Run Locally

### 1. Backend

From the project root, activate your existing Python environment and install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\backend\requirements.txt
```

Start the API:

```powershell
python -m uvicorn backend.main:app --reload
```

The API is expected to run at `http://127.0.0.1:8000` with interactive documentation at `http://127.0.0.1:8000/docs`.

### 2. Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

## Important Notes

- Keep the trained model and its class-name JSON file at the paths expected by the backend.
- The dataset and local SQLite databases are not included in the repository.
- This project is for educational and research purposes only. It is not a substitute for professional medical evaluation.
- Confirm backend CORS settings, model paths, and production environment configuration before deployment.

## License

No license has been specified yet. Contact the repository owner before reusing or redistributing this project.
