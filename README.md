# Automated Photo Tagging & Organization in Social Media Platforms  

> Revolutionizing how users manage and explore their digital memories.  

---

## 📌 Overview  

Social media users upload millions of photos daily, but **manual tagging** of friends and family is inefficient and often incomplete. Our project leverages **Machine Learning (Haar Cascade + PCA + SVM)** to automatically **detect, recognize, and tag faces** in uploaded photos.  

This system reduces the burden of manual tagging, improves user experience, and enables faster organization of digital photo libraries.  

---

## 🎯 Problem Domain: Social Media & Digital Content Management  

1. **Manual Tagging Inefficiency**  
   - Users spend significant time tagging photos.  
   - Many photos remain untagged → disorganized galleries.  

2. **Automated Solution Benefits**  
   - Instantly find and tag people.  
   - Encourage more interaction on the platform.  
   - Save user time and effort.  

3. **Broader Applications**  
   - Beyond social media → can be used in personal photo libraries, event albums, or enterprise image management systems.  

---

## 🗂️ Dataset: Labeled Faces in the Wild (LFW)  

We selected the **LFW dataset**, a robust benchmark for face recognition research, well-suited to our real-world problem.  

- **Scale:** Over **13,000 labeled face images**.  
- **Diversity:** **1,680 unique individuals** with multiple images each.  
- **Realism:** Photos include variations in pose, lighting, and background.  
- **Source:** [Kaggle – LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)  

### ✅ Why LFW?  

- **Real-World Variability** → Mimics typical social media uploads.  
- **Large & Diverse** → Sufficient for model training and testing.  
- **Multiple Images per Person** → Helps robust identification.  
- **Public & Well-Labeled** → Perfect for supervised learning.  

**Input Features:** Pixel values (grayscale face images).  
**Target Label:** Person’s name (encoded into numeric form).  

---

## 🔎 Data Preprocessing & EDA (Team Roles)

Data preprocessing and exploratory data analysis (EDA) are **essential steps** in preparing our dataset for machine learning.  
Each group member handled **one unique preprocessing step**, ensuring a clean and optimized dataset for training our face recognition model.  

---

### 👤 Member 1 : IT24102240 – Encoding Categorical Variables
- **Problem:** The LFW dataset provides labels as text (e.g., `"person_1"`, `"person_2"`). Machine learning algorithms like SVM cannot work directly with text.  
- **Solution:** We applied **Label Encoding** to convert text labels into numerical values (e.g., `"person_1" → 0`, `"person_2" → 1`).  
- **Why Important:** Enables the model to learn class mappings in a numerical format.  
- **Visualization:** Bar chart showing **class distribution** (how many images per person). Helps detect if the dataset is **imbalanced**.

---

### 👤 Member 2 : IT24102217 – Face Detection & Cleaning  
- **Problem:** Dataset contained **unreadable images** and **photos without detectable faces**, introducing noise.  
- **Solution:**  
  - Skipped unreadable images.  
  - Converted images to **grayscale** for consistent detection.  
  - Used **Haar Cascade** to detect faces.  
  - Cropped detected face regions and **resized to 50×50**.  
  - Added only **valid face images** to the dataset.  
- **Why Important:** Ensures the dataset has **clean, standardized face-only data**, improving training reliability and reducing errors.  
- **Visualization:** Bar chart showing **valid faces collected** vs. **skipped images** (due to load error / no face).  

--- 

---

### 👤 Member 3 : IT24102314 – Outlier Detection / Removal
- **Problem:** Some detected faces may still be **too dark, too bright, or unusual**.  
- **Solution:**  
  - Calculated **mean pixel intensity** and **variance** for each image.  
  - Used a **boxplot** to find outliers (faces with extreme brightness values).  
  - Removed these outliers from the dataset.  
- **Why Important:** Outliers confuse the classifier and can reduce generalization performance.  
- **Visualization:** Boxplot of average pixel intensity (outliers appear as points outside whiskers).  

---

### 👤 Member 4 : IT24102267 – Normalization / Scaling (StandardScaler)
- **Problem:** Raw pixel values range from **0–255**. Features with large ranges can dominate, making it difficult for algorithms like SVM and PCA to perform well.  
- **Solution:** Applied **StandardScaler** from scikit-learn:  
  - Transforms features so that each has **mean = 0** and **standard deviation = 1**.  
- **Why Important:**  
  - Ensures all pixel features contribute **equally**.  
  - Prevents bias toward high-value pixels.  
  - Improves convergence of SVM and PCA.  
- **Visualization:** Histograms of pixel intensity **before and after scaling** show how the distribution becomes centered around 0.  

---

### 👤 Member 5 : IT24102313 –  Feature Engineering (PCA – Dimensionality Reduction)
- **Problem:** Each face image (50×50) results in **2500 features**, which is very high-dimensional. High-dimensional data causes longer training time and risk of overfitting.  
- **Solution:**  
  - Applied **Principal Component Analysis (PCA)** to reduce features (e.g., 2500 → 100).  
  - PCA retains maximum variance while reducing dimensions.  
- **Why Important:**  
  - Reduces noise.  
  - Speeds up training.  
  - Improves accuracy by focusing on the most informative features.  
- **Visualization:**  
  - **Scree plot** (variance explained by each component).  
  - **2D scatter plot** of first two PCA components showing separation between classes.  

---

### 👤 Member 6 : IT24102310 – Feature Selection / Correlation Analysis
- **Problem:** Some pixel features are redundant (e.g., background pixels with very little variance).  
- **Solution:**  
  - Applied **VarianceThreshold** to remove low-variance features.  
  - Optionally checked for highly correlated features and removed redundancy.  
- **Why Important:**  
  - Reduces dataset size further.  
  - Keeps only useful features for recognition.  
  - Prevents overfitting.  
- **Visualization:** Variance distribution plot (before vs after feature selection).  

---

## ⚙️ How to Run the Code (Data Preprocessing)

Our preprocessing pipeline was implemented in **Jupyter Notebook** for better visualization and step-by-step explanation.  
Follow these steps to reproduce our preprocessing results and generate the final dataset (`data.npy` and `target.npy`).

### 1️⃣ Open Jupyter Notebook
Navigate to the project folder and launch Jupyter

### 2️⃣ In the Jupyter interface, open:
group_pipeline.ipynb

### 3️⃣ Run All Cells
Execute the notebook cells in sequence (Shift + Enter).

---

### ✅ Final Output
After completing all preprocessing steps, we saved:  
- `data.npy` → Processed face features (scaled, cleaned, and reduced).  
- `target.npy` → Encoded labels (numeric categories for each person).  

These files are used in training the **Support Vector Machine ** for face recognition
