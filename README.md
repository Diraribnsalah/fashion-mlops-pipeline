#  FashionMNIST Production MLOps Pipeline

An end-to-end Machine Learning pipeline for FashionMNIST image classification. This project demonstrates MLOps practices across model training with PyTorch, model export to ONNX, optimized inference with ONNX Runtime, containerization with Docker, a Gradio web interface, and automated CI/CD with GitHub Actions.

---

##  Table of Contents

- [Architecture & Workflow](#-architecture--workflow)
- [Key Features](#-key-features)
- [Project Directory Structure](#-project-directory-structure)
- [Prerequisites](#-prerequisites)
- [Local Setup & Execution](#-local-setup--execution)
  - [Method 1: Running with Native Python](#method-1-running-with-native-python)
  - [Method 2: Running with Docker](#method-2-running-with-docker)
- [CI/CD & Docker Hub Automation](#-cicd--docker-hub-automation)
- [Model Evaluation & Artifacts](#-model-evaluation--artifacts)
- [MLOps Best Practices Demonstrated](#-mlops-best-practices-demonstrated)

---

##  Architecture & Workflow

```text
[PyTorch Model Training]
          |
          v
   [Export to ONNX]
          |
          v
[ONNX Runtime Inference Engine]
          |
          v
 [Gradio Web Interface (app.py)]
          |
          v
 [Docker Containerization]
          |
          v
 [GitHub Actions CI/CD]
          |
          v
 [Automated Push to Docker Hub]
```

##  Key Features

- **Model Training & Optimization:** A custom Convolutional Neural Network (CNN) trained with PyTorch and exported to ONNX for inference using ONNX Runtime.
- **Serving & User Interface:** A web application built with Gradio (`app.py`) that uses `model_inference.py` to make predictions.
- **Modular Codebase:** Separate modules for data loading (`dataset.py`), model architecture (`model.py`), evaluation (`evaluate.py`), and inference (`model_inference.py`).
- **Containerization:** Docker packages the application and its dependencies into a consistent runtime environment.
- **CI/CD Automation:** GitHub Actions can automate building and publishing the Docker image to Docker Hub when configured.
- **Deployment-Ready Structure:** The app can be adapted for deployment to Hugging Face Spaces.

##  Project Directory Structure

The project structure is shown below.

```text
 Fashion-MLOps-Pipeline/
├── .github/
│   └── workflows/
│       └── deploy.yml          # GitHub Actions CI/CD pipeline
├── data/                       # Dataset storage (excluded via .gitignore)
├── .dockerignore               # Docker build exclusions
├── .gitignore                  # Git tracking exclusions
├── Dockerfile                  # Production Docker container build file
├── app.py                      # Main Gradio application entry point
├── model_inference.py          # Preprocessing logic & ONNX execution engine
├── model.py                    # PyTorch CNN model architecture
├── dataset.py                  # Data loading and transformation pipeline
├── evaluate.py                 # Evaluation script for generating metrics
├── fashion_cnn.onnx            # Optimized ONNX model weights
├── fashion_cnn.onnx.data       # Supplementary ONNX model binary data
├── fashion_cnn.pth             # Trained PyTorch checkpoint
├── confusion_matrix.png        # Generated evaluation metric visual
├── requirements.txt            # Production Python dependencies
└── README.md                   # Complete project documentation
```

 

##  Prerequisites

- **Python:** 3.10 or later
- **Docker Engine:** Installed and running (for the Docker method)
- **Git:** Installed on your local machine
- **Docker Hub account:** Required if you want to publish the image to Docker Hub

##  Local Setup & Execution

### Method 1: Running with Native Python

1. **Clone the repository**

   Replace `YOUR_USERNAME` with your GitHub username and update the repository URL if needed.

   ```bash
   git clone  https://github.com/Diraribnsalah/Fashion-MLOps-Pipeline.git
   cd Fashion-MLOps-Pipeline
   ```

2. **Create and activate a virtual environment**

   **Linux / macOS**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows (PowerShell)**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   **Windows (Command Prompt)**
   ```bat
   python -m venv .venv
   .venv\Scripts\activate.bat
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the application**

   ```bash
   python app.py
   ```

5. **Open the web interface**

   Visit [http://localhost:7860](http://localhost:7860) in your browser, provided the app is configured to run on port `7860`.

### Method 2: Running with Docker

Make sure Docker is installed and running, and that the repository includes a valid `Dockerfile`.

1. **Build the Docker image**

   ```bash
   docker build -t fashion-mlops-app .
   ```

2. **Run the container**

   ```bash
   docker run --rm -p 7860:7860 fashion-mlops-app
   ```

3. **Open the web interface**

   Visit [http://localhost:7860](http://localhost:7860) in your browser.

> **Note:** The Dockerfile and Gradio app must be configured to listen on port `7860` and bind to `0.0.0.0` inside the container for port forwarding to work.

##  CI/CD & Docker Hub Automation

GitHub Actions can build the Docker image and push it to Docker Hub automatically when a workflow is configured.

Typical setup:

1. Create a Docker Hub access token.
2. In your GitHub repository, open **Settings → Secrets and variables → Actions**.
3. Add the secrets expected by your workflow, commonly:
   - `DOCKERHUB_USERNAME`
   - `DOCKERHUB_TOKEN`
4. Confirm that the workflow file under `.github/workflows/` uses the same secret names and the correct Docker Hub repository.
5. Push a commit to the branch or trigger configured by the workflow.

The exact trigger, image tag, and secret names depend on your workflow implementation.

 

 

 

##  MLOps Best Practices Demonstrated

- **Modularity:** Separating training, evaluation, inference, and UI code.
- **Model interoperability:** Exporting a PyTorch model to ONNX.
- **Optimized inference:** Using ONNX Runtime for model execution.
- **Reproducible environments:** Managing dependencies and packaging the application with Docker.
- **Automation:** Using GitHub Actions for repeatable build and publishing workflows.
- **Deployment:** Providing a Gradio interface that can be hosted locally or adapted for a deployment platform.

---

## License

Add your chosen license and include a `LICENSE` file in the repository if you intend to distribute this project publicly.
