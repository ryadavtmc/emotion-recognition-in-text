# Emotion Recognition in Text Using NLP

This project focuses on emotion classification in social media text using traditional machine learning techniques. The goal is to classify each text snippet into one of six Ekman emotions: **joy**, **anger**, **sadness**, **fear**, **disgust**, or **surprise**.

---

## 🧠 Project Highlights

- **Dataset**: Google’s [GoEmotions dataset](https://aclanthology.org/2020.coling-main.372) with 58 fine-grained emotions was mapped to 6 core Ekman emotion categories.
- **ML Techniques**: Logistic Regression, SVM, Random Forest, XGBoost, Voting Ensembles, and Stacking Classifiers.
- **No Deep Learning**: Designed to work on modest hardware using scikit-learn & XGBoost.

---

## 📁 Data & Preprocessing

- **Source Files**:
  - `train.tsv`, `dev.tsv`, `test.tsv`: Raw GoEmotions dataset splits.
  - `emotions.txt`: List of emotion labels.
  - `ekman_mapping.json`: JSON mapping of 58 GoEmotions labels to Ekman’s six core emotions.

- **Steps**:
  - Load and merge datasets.
  - Map fine-grained labels to Ekman classes.
  - Drop samples with only "neutral".
  - Encode target labels.
  - Use TF-IDF with unigram + bigram features (max_features=10,000).

---

## 🔍 Feature Engineering

- **TF-IDF Vectorization**: Convert raw text into a sparse matrix of token importance using unigrams and bigrams.
- **Label Encoding**: Categorical emotion labels were encoded into numeric form for classification.

---

## 🧪 Models Used

| Model                  | Description |
|------------------------|-------------|
| **Logistic Regression** | Linear baseline with class balancing. |
| **Support Vector Machine** | Linear kernel with probability output. |
| **Random Forest** | Ensemble of decision trees with class weights. |
| **XGBoost** | Boosted trees with `mlogloss` objective. |
| **Voting Classifier (Soft)** | Combines predictions of LR, RF, and XGB. |
| **Stacking Classifier** | Meta-model on top of base learners. Tested Ridge, Logistic Regression, Random Forest, and XGB as meta models. |

---

## 📈 Evaluation Metrics

- **Precision, Recall, F1-score** (per class and macro-average)
- **Confusion Matrix**: Visual breakdown of misclassifications.
- **F1-score Comparison Charts**: Bar charts to compare model performance on each emotion.

---

## 🏆 Key Findings

- **Class Imbalance**: Joy had the most samples (~21k), fear/disgust were rare (<1k).
- **Logistic Regression**: Performed best on frequent emotions (Joy: 0.826 F1).
- **Voting Classifier**: Delivered best overall performance, boosting F1 scores for minority classes like fear and disgust.
- **Stacking**: Showed improvement in sadness and anger detection with Ridge/LogReg meta-models.
- **Misclassifications**: Common between similar emotions (e.g., joy ↔ surprise, anger ↔ sadness).

---

## 📌 Discussion

- Ensemble methods improved robustness on imbalanced data.
- TF-IDF was surprisingly effective despite its simplicity.
- No deep learning was used, making it suitable for low-resource setups (e.g., Mac mini).
- Emotion classification is still sensitive to overlapping linguistic patterns.

---

## 📚 References

- Demszky, D., et al. (2020). *GoEmotions: A dataset of fine-grained emotions*. Google Research. https://aclanthology.org/2020.coling-main.372  
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine learning in Python*. Journal of Machine Learning Research, 12, 2825–2830. https://scikit-learn.org  
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. https://doi.org/10.1145/2939672.2939785  
- *Imbalanced-learn documentation*. https://imbalanced-learn.org  

---

## 💻 Requirements

- Python 3.9+
- pandas, scikit-learn, seaborn, matplotlib
- xgboost