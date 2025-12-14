# Architectural Floor Plan Object Detection System

This is a full-stack object detection system based on the PyTorch Faster R-CNN model, used to automatically identify key elements in architectural floor plans, such as **walls** and **rooms**.

This system implements an end-to-end solution, from deep learning model training to high-performance API services and a user-friendly web interface.

## Key Features

**Model Efficiency:** The **PyTorch Faster R-CNN (ResNet50-FPN)** model was fine-tuned on the CubiCasa5K dataset using **transfer learning**.
* **Training optimizations:** Supports **FP16 half-precision training** (via `torch.cuda.amp`) and a **Cosine Annealing** learning rate scheduler, ensuring fast and resource-efficient training.
* **High-performance service**: Built on the **FastAPI** framework, this asynchronous backend provides a low-latency, real-time inference API and supports **CORS** cross-domain access.
* **Full-stack delivery:** Provides a user-friendly HTML/JavaScript front-end interface, supporting drag-and-drop uploads, dynamic adjustment of confidence thresholds, and visualization of detection results.
* **Deployment Automation**: Provides a one-click startup script (`start.py` or `START_BACKEND.bat`) to automatically check dependencies, load models, and **concurrently start** front-end and back-end services.
* **Flexible Inference:** Provides a standalone inference script (`simple_inference.py`) that supports single-image inference, batch inference, and COCO format evaluation.

## 🛠️ Technology Stack

| Field | Technology/Tools | Description |
| :---  | :--------------- | :---------- |
| Deep Learning | PyTorch, TorchVision | Core Model Training and Inference Frameworks |
| Model Architecture| Faster R-CNN (ResNet50-FPN) | A classic architecture for object detection |
| Data Science | Albumentations, PyCOCOtools | Advanced Data Augmentation and Standard Evaluation Tools |
| Backend Services | Python, FastAPI, Uvicorn | High-performance, asynchronous RESTful API framework |
| Front-end Interface | HTML5, CSS, JavaScript | Lightweight Single-Page Application (SPA) User Interface |


## 📂 File Structure and Functionality Breakdown

This project's file structure is designed to clearly separate the core deep learning components from the API serving and deployment automation layers.

| Filename | Function / Description | Key Features |
| :--- | :--- | :--- |
| `simple_train.py` | **Model Training Script** | Responsible for dataset loading, initializing the **Faster R-CNN** model, and implementing **Albumentations** data augmentation, **FP16 training**, and **COCO evaluation** logic. |
| `simple_inference.py` | **Model Inference Utility** | Provides the `DetectionInference` class for loading the model, image preprocessing, and executing single/batch inference. Includes methods for drawing visualization and COCO evaluation. |
| `app.py` | **FastAPI Backend Service** | The core service entry point, which loads the model at startup and exposes the detection endpoints. Provides `/detect` (returns JSON results) and `/detect-with-visualization` (returns drawn image file) RESTful APIs. |
| `index.html` | **Frontend User Interface** | A lightweight Single Page Application (SPA) responsible for user interaction, file upload, and calling the FastAPI API via JavaScript (Fetch) to display real-time detection results and statistics. |
| `start.py` | **Python Quick Start Script** | The unified launcher that checks dependencies and uses Python concurrency to launch the `app.py` (Uvicorn backend) and the front-end server concurrently, with automatic browser opening. |
| `START_BACKEND.bat` | **Windows Batch Script** | A convenience script for Windows users to check the environment (Conda activation) and launch both the backend and frontend services. |
| `pytorch_detection_results/` | **Output Directory** | The default directory for saving model checkpoints (`best_model.pth`) and inference results. |
