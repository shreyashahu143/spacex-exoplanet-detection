## 🚩 Problem Statement
Our initial goal was to solve the ***“cosmic haystack”***
problem: finding exoplanet “needles” hidden in vast datasets.

While AI can detect candidates, scientists still face the *slow, manual process* of verifying each one.
Our challenge was not just to find candidates, but to build an *automated machine learning pipeline* that accelerates discovery and reduces dependence on manual techniques.

---

## 📘 Prototype Overview
We developed an *automated ML pipeline* to identify exoplanets using NASA’s *Kepler* dataset.

- *Input:* astrophysical features like Orbital Period (days), Transit Duration (hours), Transit Depth (ppm), Planetary Radius (Earth radii), Equilibrium Temperature (K), Stellar Radius (Solar radii), impact & Stellar Temperature.

- *Process:* data preprocessing steps,
 1. The first step included data cleaning ,Cleaned the raw Kelper dataset csv file  followed by classification using         advance ensemble models.
 2. The dataset consist of in total
 3. The 22 attributes were used in the data preprocessing step.
 4. The 5 another attributes wee included using the feature engineering ( log transformation ,habitable zone, duration to period ratio)

- *Output:* high-confidence predictions of whether a candidate is a *confirmed exoplanet* or a *false positive*.

Our prototype aims to balance *speed, accuracy, and interpretability* — .

👉 *XGBoost* was selected as the *final model* due to its *highest accuracy* and ability to capture complex feature interactions.

---

## Model comparison
The aim of this comparison was to evaluate different algorithms based on performance, accuracy, and efficiency. Each model was trained, tested, and benchmarked to understand its strengths and weaknesses in handling the dataset.

🔍 Key Insights

Random Forest: Provides robust predictions and handles feature importance well, but may be slower with very large datasets.

Ensemble Model: Combines predictions from multiple algorithms, leading to improved overall performance and reduced variance.

XGBoost: Highly efficient and powerful, often achieving the best accuracy, but requires careful tuning of hyperparameters.

## 🔍 About the Final Model

*XGBoost Classifier (Final Choice)*

- *Why chosen:*
  - Delivered the *highest accuracy* among tested models.
  - Handles missing data and imbalanced classes effectively.
  - Captures nonlinear feature interactions that simpler models miss.
  - Scales well to large datasets such as Kepler and TESS archives.

- *Input features used:*
  - Orbital Period (koi_period)
  - Transit Depth (koi_depth)
  - Transit Duration (koi_duration)
  - Planet Radius (koi_prad)
  - Stellar Radius (koi_srad)
  - Equilibrium Temperature (koi_teq)
  - Additional engineered features (log transforms, habitable zone flag).

-  *Performance goal:*
  - Achieved Accuracy > 0.95
  - Strong Precision/Recall balance → minimizes false positives
  - Produces stable, reproducible results

---

## Video Demonstration 
- *this is the live recording of our working model - https://drive.google.com/file/d/1iTmttCQ04ChX4MDsr3fKMNha0BLiG3z1/view?usp=sharing

## ✨ Unique Features of Our Prototype

- *Automated Pipeline:* End-to-end process from data preprocessing to classification.
- *SMOTE Handling:* Balanced dataset ensures fair learning across classes.
- *Glass-Box Science:* Feature importance explains why a candidate is flagged.
- *High Accuracy with XGBoost:* Gradient boosting delivers top performance on astrophysical features.
- *Accessible & Reproducible:* Built with open-source libraries (Scikit-learn, Pandas, imbalanced-learn, XGBoost).
- *Scalable:* Can be adapted to future datasets (e.g., TESS, JWST).

## Video demostration

*this is goggle drive link which will take you to the video file of actual working of the prototype*

https://drive.google.com/file/d/1iTmttCQ04ChX4MDsr3fKMNha0BLiG3z1/view?usp=drive_link
