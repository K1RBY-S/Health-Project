# Health & Lifestyle Classification Analysis

## Overview
This project explores whether lifestyle behaviors (fast-food consumption frequency, daily calorie intake, physical activity, sleep duration, and self-reported energy levels) can be used to classify individual health outcomes using a BMI-based target variable.

The goal of the project was not to optimize for maximum model performance, but to evaluate whether commonly assumed behavioral indicators meaningfully predict health outcomes.

---

## Data
- **Source:** Kaggle (structured dataset combining synthetic and real-world patterns)
- **Observations:** Individual-level lifestyle and health attributes
- **Target Variable:** Binary BMI classification (BMI > 25)

> Note: The full dataset may not be included in this repository due to dataset licensing restrictions.  
> If needed, download the dataset from Kaggle and place it in the project root as `fast_food.csv`.

---

## Methodology
1. **Data Cleaning**
   - Removed invalid and ambiguous categorical values
   - Handled missing values
   - Encoded categorical variables

2. **Feature Engineering**
   - Created binary BMI target variable
   - Standardized numerical features
   - Applied one-hot encoding where appropriate

3. **Models Evaluated**
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier

4. **Evaluation Metrics**
   - Accuracy (train vs. test)
   - Precision, Recall, F1-score
   - ROC-AUC

---

## Results Summary
- Logistic Regression demonstrated the best generalization performance on unseen data.
- Decision Tree and Random Forest models achieved perfect training accuracy, indicating overfitting.
- ROC-AUC scores near 0.5 suggest limited separability between classes.
- Results indicate that lifestyle behavioral factors alone may be insufficient for reliably classifying individual health outcomes.

---

## Key Takeaways
- Higher model complexity did not improve generalization.
- Behavioral features showed substantial overlap across BMI classes.
- Model performance limitations are driven more by data characteristics than modeling deficiencies.

---
