
# Twitter Sentiment Analysis

This repository contains code and analysis for classifying sentiments (Positive, Neutral, and Negative) in tweets related to airlines using ensemble machine learning models and an LSTM-based deep learning model. The project focuses on building an accurate sentiment classifier while addressing the class imbalance inherent in the dataset.

---

## Project Overview

### Dataset
The dataset used in this project contains airline-related tweets, each labeled with one of three sentiment classes:
- **Positive**
- **Neutral**
- **Negative**

The dataset is imbalanced, with **Negative** tweets outnumbering **Positive** and **Neutral** ones.

### Data Distribution
- **Training set**: 8,260 negative tweets (majority class), with fewer positive and neutral tweets.
- **Validation set**: Contains a similar distribution.

### Data Preprocessing
1. **Cleaning**: 
   - Convert text to lowercase.
   - Remove stop words, punctuation, special characters, digits, URLs, and tags.
   - Expand contractions (e.g., "don't" to "do not").
   
2. **Text Vectorization**: 
   - Word embeddings were generated using **Word2Vec** to map words into vectors based on their semantic similarity.
   
3. **Padding**: 
   - Padding was applied to ensure uniform input sequence lengths for the deep learning models.

---

## Model Training

We explored several classification models, including ensemble learning models and an LSTM-based deep learning model.

### Models Used
1. **Ensemble Learning Models**:
   - **Random Forest**
   - **XGBoost**
   - **AdaBoost**
   - **Multinomial Naive Bayes (MultinomialNB)**
   - **Support Vector Machine (SVM)**
   
2. **Deep Learning Model**:
   - **LSTM** (Long Short-Term Memory)
     - The LSTM model was trained using an embedding layer, LSTM layer, dropout for regularization, and a dense layer with a softmax activation function for multi-class classification.

### Training Details:
- The **LSTM model** was trained for 50 epochs with a batch size of 1024.
- **EarlyStopping** and **ReduceLROnPlateau** were used as callbacks to optimize training, prevent overfitting, and adjust the learning rate during stalled learning.
- **Adam optimizer** was employed with a learning rate of 0.0001.

---

## Performance Evaluation

### Model Performance Comparison

| Model                | Training Accuracy | Validation Accuracy |
|----------------------|-------------------|---------------------|
| **Random Forest**     | ~74.71%           | ~75.93%             |
| **AdaBoost**          | ~72.54%           | ~73.68%             |
| **MultinomialNB**     | ~71.21%           | ~72.10%             |
| **SVM**               | ~73.40%           | ~73.45%             |
| **LSTM**              | **77.01%**        | **76.84%**          |

### Results Breakdown

- The **LSTM model** performed best with 77.01% training accuracy and 76.84% validation accuracy, outperforming all ensemble models.
- **XGBoost** and **AdaBoost** performed comparably well, but the LSTM model had superior performance in terms of both accuracy and F1-score.

### Accuracy and Loss Graphs
![Accuracy & Loss](path_to_accuracy_loss_graph.png)

### Confusion Matrix for LSTM Model
![LSTM Confusion Matrix](path_to_lstm_confusion_matrix.png)

### F1-Score Comparison
The LSTM model had the highest F1-scores for all sentiment categories:
- **Negative**: 0.76
- **Positive**: 0.37
- **Neutral**: 0.33

---

## Hyperparameter Tuning

The LSTM model underwent hyperparameter tuning to improve performance:
- **Batch size** was reduced to 256.
- **ModelCheckpoint** was added to restore the best weights during training.

### Tuning Results

| Metric            | Pre-Tuning       | Post-Tuning     |
|-------------------|------------------|-----------------|
| **Training Accuracy** | 74.71%        | 77.01%          |
| **Validation Accuracy** | 75.93%      | 76.84%          |
| **Training Loss**      | 61.69%       | 55.98%          |
| **Validation Loss**    | 60.84%       | 57.17%          |

- The F1-score of the LSTM model improved marginally post-tuning, with slight gains in positive and neutral sentiment classification.

---

## Conclusion

The **LSTM model** proved to be the best classifier for sentiment analysis in this dataset, offering high accuracy and robust performance across all sentiment categories. The model is saved as `SentimentAnalysis.h5` for deployment.

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   - Open and run `tweet_analysis.ipynb` to reproduce the results.

---

## Results and Visualizations
- All training and validation metrics, confusion matrices, and F1-scores are available in the `results` folder.
- The final trained model, `SentimentAnalysis.h5`, is included for prediction and deployment.

---

## References
- [Raschka, S. et al. "Python Machine Learning"](https://www.python-machine-learning.com)
- [Word2Vec Documentation](https://www.tensorflow.org/tutorials/text/word2vec)
- [Keras Callbacks API](https://keras.io/api/callbacks/)
