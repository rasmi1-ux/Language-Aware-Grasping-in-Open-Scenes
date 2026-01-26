# Open World Grasping (OWG) Pipeline
The ability to grasp objects in-the-wild from open-ended language instructions constitutes a fundamental challenge in robotics. We propose OWG, an open-world grasping pipeline that combines SAM 2.1 and Qwen-VL to unlock grounded world understanding in three stages: open-ended referring segmentation, physical reasoning for multi-step planning (e.g., removing blockers before grasping), and grasp ranking via VLM-based evaluation. This approach enables the system to operate zero-shot via visual prompting, allowing a Franka Emika Panda robot to perceive, reason, and act in complex, unstructured environments without task-specific training.

## 📖 Overview

This repository contains an **open-world grasping pipeline** that:
* **Segments** objects in an RGB image (SAM2 automatic mask generation).
* **Grounds** a natural-language query to a target instance ID (Qwen3‑VL).
* **Plans** a high-level action (e.g., `pick` / `remove`) based on physical reasoning (Qwen3‑VL).
* **Generates** geometry-based grasp candidates and optionally **ranks** them with the VLM.
* **Executes** the grasp on a **Franka Panda** using MoveIt 2 and the gripper action interface.

## 🚀 Features

* **Natural Language Grounding:** Uses **Qwen3-VL** to understand complex user queries (e.g., *"pick the red block on top of the book"*) and identify target objects in the scene.
* **Zero-Shot Segmentation:** Leverages **SAM 2.1 (Segment Anything Model)** to segment novel objects in an open-world setting without prior training on specific classes.
* **Physical Reasoning:** The system detects stacking and occlusion relationships. If a target is covered, it generates a multi-step plan to `remove` the blocking object before picking the target.
* **Hybrid Grasping:** Combines geometric grasp generation (contact-based) with VLM-based semantic ranking to select the most stable and collision-free grasp.
* **ROS 2 Integration:** Fully integrated with **ROS 2** (Humble/Jazzy) and **MoveIt 2**, featuring a modular architecture with separate nodes for perception and robot control.

* ## 🛠️ Prerequisites

### Hardware
* **Robot:** Franka Emika Panda.
* **Camera:** RGB-D Camera.
  * *Supported:* Orbbec Femto Bolt or Intel RealSense.
  * *Mounting:* Camera must be hand-mounted or fixed relative to the robot base.
    
### Software Dependencies
* **ROS 2:** Jazzy.
* **Hardware Interface:** The `mul_franka` repository installed in your workspace.

## 🔧 Robot & Hardware Setup

Before running the OWG pipeline, you must set up the `mul_franka` drivers and ensure the real-time kernel is active.

### Check & Install the [`mul_franka`](https://github.com/mul-cps/mul_franka) Workspace

## 📦 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/owg.git
cd owg
```

### 2. Set Up Python Environment (Recommended)

Using **Conda** is strongly recommended to manage dependencies and avoid conflicts.

```bash
conda create -n owg python=3.12.12 -y
conda activate owg
```

### 3. Install Core Libraries

Install the required Python packages from the requirements.txt file.
```bash
pip install -r requirements.txt
```

### 4. Install SAM 2 (Segment Anything 2.1)

**Download the Model Checkpoint:** Download the specific SAM 2.1 model weights into the root directory. We default to Base Plus.

* **Base Plus (Default):**
    
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
```
* **Large:**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```
* **Small:**
```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```
**Note: If you use the Large or Small model, you must update the configuration in pipeline.py:**
```bash
    # Example for Large
    sam_config="configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_checkpoint="sam2.1_hiera_large.pt"
```
For full documentation, visit [*PyPI - SAM 2*](https://pypi.org/project/sam2/).

## 5. Install Qwen (Vision-Language Model)

You need to download the **Qwen3-VL-4B-Instruct** model and place it inside a folder named Qwen.

* **Option A: Using Git LFS (Recommended)**
```bash
# Make sure you have git-lfs installed
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct Qwen
```
* **Option B: Manual Download**
* Download all files (safetensors, config, tokenizer) from [*Hugging Face*](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/tree/main) and place them in a folder named Qwen/ in the project root.

## 6. Final Directory Structure

After installation, your directory should look like this:

```plaintext
.
├── franka_example.py
├── Grasp_detector.py
├── model.py
├── pipeline.py
├── reader.py
├── requirements.txt
├── sam2.1_hiera_base_plus.pt
├── segmentor.py
├── prompts/
│   ├── grasp_planning.txt
│   ├── grasp_ranking.txt
│   └── referring_segmentation.txt
└── Qwen/
    ├── chat_template.json
    ├── config.json
    ├── generation_config.json
    ├── merges.txt
    ├── model-00001-of-00002.safetensors
    ├── model-00002-of-00002.safetensors
    ├── model.safetensors.index.json
    ├── preprocessor_config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── video_preprocessor_config.json
    └── vocab.json

 
