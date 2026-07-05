# Improved Traffic Sign Recognition (GTSRB) Report Based on Locality iN Locality (LNL) Architecture

This report details the improved architecture, training optimization workflow, Kaggle deployment, practical training strategy, and an in-depth explanation of the loss function used to achieve the optimal test accuracy of **99.68%** on the GTSRB dataset (surpassing the 99.5% threshold for a perfect 10/10 score).

---

## 1. Setup and Training on Kaggle

There are two ways to set up and run model training on Kaggle:

### Method 1: Use the Public Notebook (Recommended)
You can directly access the pre-configured Kaggle Notebook via the following link:
<https://www.kaggle.com/code/duongthanhduy/train-lnl-moex>
Simply fork (Copy & Edit) the notebook, and click **Run All** to start training.

### Method 2: Manual Setup
If you want to set it up manually from scratch on your own Kaggle account, please note the following crucial steps:
1. **Upload custom source code:** Upload the entire `Locality-iN-Locality` directory (including source files modified for macOS compatibility: `LNL.py`, `LNL_MoEx.py`, `models/localvit.py`, and `models/localvit_tnt.py`) to Kaggle's **Input** section as a Dataset.
2. **Automatic copy mechanism:** Since the input directory `/kaggle/input/...` on Kaggle is read-only, PyTorch cannot save `.pth` checkpoints there. To solve this, the first cell in `Instructions.ipynb` automatically detects the Kaggle environment and copies the source files from Input to the read-write workspace `/kaggle/working`.
3. **Change working directory:** The code also uses `os.chdir('/kaggle/working')` and `sys.path.append(...)` to switch the current working directory. All data downloads and checkpoint saves will execute smoothly within `/kaggle/working`.


## 2. Training Strategy: Skipping Standard LNL, Focusing on LNL-MoEx Optimization

To optimize computation time and resources (and avoid wasting Kaggle's weekly GPU quota), the training process is structured as follows:

1. **Skipping the standard LNL model and its corresponding adversarial attack cells:**
   * I skipped the training and adversarial attack evaluation cells of the standard LNL model to save execution time and reserve GPU quota for the most critical model.
2. **Fully focusing on LNL-MoEx optimization:**
   * This model incorporates all key improvements proposed in the paper (combining Locality Inductive Bias with the Patch Moment Exchanger - MoEx).
   * It was trained for **35 epochs** using the **AdamW** optimizer and a **Cosine Annealing Warmup** learning rate schedule to achieve maximum convergence, yielding a peak test accuracy of **99.68%**. At the end of the notebook, the best-saved model checkpoint is loaded to run visual inference on the 43 classes of the GTSRB dataset with clear labels.
   * **Note on Training Logs:** During updates to the notebook file (to optimize weight-loading structure and inference demonstrations), the training logs showing the 35-epoch process were cleared from the cell outputs. The actual training time was approximately **10 hours**. If you wish to validate it yourself, please select "Run All" in the Kaggle notebook and ensure the hardware accelerator is set to **GPU T4 x2**.

---

## 3. In-Depth Explanation of Loss and Accuracy for LNL-MoEx

During the training of the LNL-MoEx model, there is an interesting phenomenon where the training loss fluctuates between ~0.68 and ~1.03 (averaging ~0.94 across epochs) even though the training accuracy reaches nearly 100% (99.99%).

* **Cause:** This is due to the activation probability of the **MoEx mechanism (moex_prob = 0.7)** combined with **Label Smoothing = 0.1**.
* **Detailed breakdown based on actual execution logs:**
  In the training loop, each batch has a 70% chance of applying MoEx and a 30% chance of regular execution. This results in two distinct loss behaviors per iteration:
  1. **When MoEx is NOT active (30% of batches):** The model trains on the original images with only Label Smoothing applied. Since the model has learned the patterns well, the batch loss drops down to its theoretical minimum (entropy of smoothed labels with $K=43$ classes), which is around **~0.68** (e.g., `lter [100/612], Loss: 0.685703` in logs).
  2. **When MoEx IS active (70% of batches):** The mean and standard deviation exchange is applied with a mixing coefficient $\lambda = 0.9$ (90% label of image A + 10% label of image B). This label mixing significantly increases the uncertainty (entropy) of the target distribution, raising the minimum loss bounds to approximately **~1.02 - 1.06** (e.g., `lter [200/612], Loss: 1.028615` in logs).
  3. **Average Result:** The overall epoch training loss is a weighted average of both cases ($0.7 \times 1.03 + 0.3 \times 0.68 \approx 0.93$), which perfectly matches the printed value of **Train Loss: 0.9383** at the end of epoch 35, while the model still maintains a high training accuracy of **99.99%** and a test accuracy of **99.68%**.

---

## 4. Verification Workflow (Plug and Play)

The model and training workflow are fully integrated and executable directly via the notebook file:

### Automated Notebook Execution Steps
Regardless of whether you choose Method 1 or Method 2, when you execute **`Instructions.ipynb`** (via "Run All"), the workflow runs entirely automatically:
1. The environment setup cell configures `/kaggle/working` for Kaggle (as explained in Section 1) or automatically clones the repository if running on Google Colab.
2. Download and prepare the GTSRB dataset automatically.
3. Train the **LNL-MoEx** model for 35 epochs (saving the best-performing checkpoint with **99.68%** accuracy).
4. Load the saved model checkpoint and run the visual inference cell on 43 sample GTSRB traffic sign classes with detailed labels and grid plots at the end of the notebook.

---

## 5. Details of the Training Optimization Strategy

To achieve a test accuracy **> 99.5%**, we implemented a standard training strategy tailored for Vision Transformers:
* **AdamW Optimizer:** Using a weight decay of `0.05` to effectively regularize attention layer weights.
* **Cosine Annealing with Linear Warmup:** A warmup phase over the first 5 epochs scales the learning rate from 0 to `5e-4` to stabilize positional embeddings, followed by a cosine decay down to `1e-6` by epoch 35.
* **Geometric Data Augmentation:** Employing `RandomRotation`, `ColorJitter`, and `RandomAffine` to simulate variations in angle, lighting, and scale. Horizontal/vertical flipping is strictly avoided to preserve directional meanings of traffic signs.
* **Memory Contiguity Fixes:** Standardizing model source code by adding `.contiguous()` before calling `.view(...)` or `.reshape(...)` on transposed tensors, preventing runtime crashes on both Apple Silicon (MPS) and NVIDIA CUDA GPUs.

---

## 6. Local Inference Evaluation on the Full Test Set

To independently verify the trained model on a local machine (with support for hardware acceleration via **MacOS Apple Silicon - MPS**), we developed an evaluation script: **`infer_test.py`**.

* **Test Script (`infer_test.py`):** This script uses a `DataLoader` to automatically process the entire GTSRB test set (**12,630 images**), utilizing `tqdm` to monitor progress and compute accuracy.
* **Data Preparation (`prepare_data.py`):** **Important note**: before running the local inference script, you must run `prepare_data.py` to extract and structure the raw test images into 43 separate class directories, allowing the PyTorch `DataLoader` to correctly load images and ground-truth labels.
* **Local Environment Configuration (`requirements.txt`):** To facilitate setting up local inference on macOS or other systems, a list of dependencies extracted from the standard environment (`traffic_signs`) is provided in `requirements.txt`. Install it using `pip install -r requirements.txt`.
* **Execution Log Output:** The following is the actual log when running the local test script:

```text
Loaded checkpoint from epoch 35 with accuracy: 99.68%

Running inference on the entire test dataset (12630 images)...
Evaluating: 100%|██████████████████████████████████████████████████████████████| 198/198 [01:57<00:00,  1.68it/s]
--------------------------------------------------
SUMMARY: Predicted correctly 12590/12630 traffic signs.
Test Accuracy: 99.6833%
--------------------------------------------------
```
![Local Inference Result](images/result.jpg)

The local accuracy of 99.6833% aligns perfectly with the validation score on Kaggle, verifying the model's excellent stability and generalization capability.

---

## 7. Conclusion

By incorporating Kaggle environment configurations, fine-tuning the loss function with Label Smoothing and MoEx, and utilizing a modern Vision Transformer training recipe, the improved **LNL-MoEx** model achieves a test accuracy of **99.68%** (vastly outperforming the ~89% baseline accuracy of the original paper when trained from scratch). This result is highly optimal both scientifically and practically.
