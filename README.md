# Hackathon
**#TCS Hackathon**

### Objective
The objective of this project is to develop a classification model to predict the credit risk of individuals. However, the dataset used did not contain a predefined target variable indicating risk categories. To overcome this, an unsupervised learning technique (K-Means clustering) was employed to create synthetic class labels. These labels were then used in a supervised classification task using a Random Forest model to identify patterns and classify new applicants accordingly.

### Problem Approach

Since the dataset did not include a predefined target variable required for classification tasks, I adopted an unsupervised-to-supervised learning pipeline to construct a meaningful target. The approach involved using K-Means clustering to generate pseudo-labels, which were then used as the target variable for training a classification model. The complete workflow is outlined below:

1. **Data Loading and Exploration**  
   The dataset was loaded and initially explored to understand its structure. This included reviewing feature descriptions, checking for missing values, and assessing data types.

2. **Data Cleaning and Preprocessing**  
   - Missing values were identified and imputed appropriately.  
   - Categorical variables were encoded using suitable encoding techniques.  
   - Outliers were detected using the Interquartile Range (IQR) method and handled only if present.

3. **Feature Engineering**  
   - Created a new feature: *Credit per Month*, derived from the ratio of credit amount to duration.  
   - Introduced a binary feature indicating whether an individual is considered *young* based on their age.

4. **Feature Scaling**  
   Applied MinMaxScaler to normalize the numerical features, ensuring all values are within a similar range for clustering and model training.

5. **Clustering to Generate Target Labels**  
   - Employed K-Means clustering to group similar data points.  
   - Reduced dimensionality to 2 components using Principal Component Analysis (PCA) for cluster visualization.  
   - The clustering results were evaluated using **Silhouette Score** (0.4225) and **Davies-Bouldin Index** (1.0340), indicating reasonable cluster separation.

6. **Target Construction**  
   The cluster assignments from K-Means were used as pseudo-labels, effectively forming a synthetic target variable for the classification task.

7. **Train-Test Split and Handling Class Imbalance**  
   The dataset was split into training and testing sets. Class imbalance was addressed using **SMOTE (Synthetic Minority Oversampling Technique)** to balance the training data.

8. **Model Training and Evaluation**  
   - A **Random Forest Classifier** was trained on the resampled data.  
   - The model was evaluated on the test set using standard classification metrics such as accuracy, precision, recall, F1-score, and ROC curve.

9. **Visualization**  
   The Receiver Operating Characteristic (ROC) curve was plotted to assess the model’s discriminatory power between the generated classes.


### Summary
This project showcases how to transform an unlabeled dataset into a classification problem using clustering-based label generation. By combining unsupervised learning for label creation with supervised learning for classification, this approach allows for effective risk prediction even in the absence of labeled target data. The methodology is scalable and can be adapted for other similar use cases in domains like finance, marketing, or healthcare.
