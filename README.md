# Sentiment Analysis: Machine Learning vs Deep Learning on IMDb Reviews

## Project Overview
This repository contains the implementation and findings of a comparative study between traditional machine learning and advanced deep learning techniques for sentiment analysis using the IMDb movie review dataset.

## Research Summary
Our study evaluates seven different models including:
- Traditional Machine Learning approaches:
  - Multinomial Naive Bayes
  - Logistic Regression
  - Gradient Boosting

- Deep Learning architectures:
  - Long Short-Term Memory (LSTM)
  - Bidirectional LSTM (BiLSTM)
  - A novel hybrid CNN-BiLSTM-Transformer model

The evaluation compares these models using metrics such as Mean Squared Error (MSE), R² score, and AUC-ROC, providing insights into their strengths and limitations for sentiment classification tasks.

## Key Findings
- **Machine Learning Models**: Gradient Boosting performed best among traditional ML approaches, with good handling of non-linear relationships.
- **Deep Learning Models**: Significantly outperformed ML models, with the hybrid CNN-BiLSTM-Transformer achieving:
  - Highest accuracy (94%)
  - Lowest MSE (0.06)
  - Highest AUC-ROC (0.96)
- **Trade-offs**: ML models offer efficiency and interpretability, while DL models excel at capturing textual complexity.

## Dataset
The IMDb movie reviews dataset contains 50,000 reviews labeled as positive or negative, making it an ideal benchmark for evaluating sentiment analysis techniques.

## Methodology
Our approach included:
1. **Data Preprocessing**:
   - Text normalization (lowercase conversion)
   - Removal of non-alphabetic characters
   - Stop word removal using NLTK
   - Lemmatization with WordNet Lemmatizer
   - Binary sentiment encoding (1 for positive, 0 for negative)

2. **Feature Extraction**:
   - Term Frequency-Inverse Document Frequency (TF-IDF)
   - Count Vectorization

3. **Model Implementation**:
   - Traditional ML models implemented with scikit-learn
   - Deep learning models built with TensorFlow/Keras

4. **Evaluation Metrics**:
   - Mean Squared Error (MSE)
   - R² Score
   - AUC-ROC curves
   - Confusion matrices

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Matplotlib
- Seaborn

## Model Performance Summary

| Model                    | R² Score | MSE   |
|--------------------------|----------|-------|
| Logistic Regression      | 0.106    | 0.575 |
| Naive Bayes              | 0.136    | 0.454 |
| Gradient Boosting        | 0.189    | 0.243 |
| LSTM                     | 0.491    | 0.127 |
| BiLSTM                   | 0.442    | 0.234 |
| CNN-BiLSTM-Transformer   | 0.490    | 0.126 |

## Conclusion
Our study demonstrates that while traditional machine learning models are efficient and interpretable, deep learning models, especially hybrid architectures like CNN-BiLSTM-Transformer, significantly outperform them in capturing the complexity of textual data for sentiment analysis.

The choice of model should align with the complexity of the dataset and the specific requirements of the task. For computationally constrained environments or simpler tasks, ML models like Gradient Boosting are suitable. For complex tasks where accuracy is paramount, deep learning models, particularly hybrid architectures, are recommended.

## Contributors
- [Rishi Akkala](https://github.com/rishiakkala)
- [Mokshyagna Kalyan](https://github.com/AMKalyan)

## Acknowledgments
- This research builds upon prior work in sentiment analysis, particularly the studies cited in our references section.
- Thanks to the creators of the IMDb movie reviews dataset for providing this valuable resource for sentiment analysis research.
