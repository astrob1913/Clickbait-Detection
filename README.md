# CLICKBAIT DETECTION


- Project Overview: 

This project investigates automated clickbait detection using Natural Language Processing (NLP) techniques. The task is formulated as a binary text classification problem, where headlines are classified as either clickbait or non-clickbait.

The project compares both traditional machine learning and deep learning approaches, including:

1. TF-IDF + Support Vector Machine (SVM)
2. Word2Vec + Support Vector Machine (SVM)
3. Long Short-Term Memory (LSTM) neural network
4. Optimiser comparison between Adam and AdamW for LSTM

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrices
- Manual error analysis

The best-performing model was the LSTM model trained using the Adam optimiser.

- Google Drive Link:


Google Drive folder containing the complete project files with the training notebook ("NLP CW_main,ipynb") and all the trained models:

https://drive.google.com/drive/folders/1QvWW3UQhGyw-5L28gFEbcuSa_saNKDpT?usp=sharing


- Project Structure:


Dataset/
    clickbait_data.csv

data/
    train.csv
    test.csv

models/
    lstm_adam.h5
    svm_tfidf_pipeline.pkl
    tokenizer.pkl

Result data/
    lstm_predictions.csv
    svm_tfidf_predictions.csv

Test_evaluation.ipynb - Here, the best two models are being evaluated on test set.


- Required Software: 


The project was developed and tested using:

- Python 3.11+
- Jupyter Notebook / Google Colab

Recommended operating systems:

- Windows 10/11
- macOS
- Linux

GPU acceleration is optional but recommended for faster LSTM training.


- Required Libraries:

Install the following libraries before running the notebooks:

pip install numpy pandas scikit-learn matplotlib tensorflow gensim nltk seaborn jupyter


- Running the Project:


The assessors only need to run:

Test_evaluation.ipynb

This notebook:

- Loads the two best-performing trained models:
    - TF-IDF + SVM
    - LSTM (Adam)

- Loads the saved preprocessing components:
    - tokenizer
    - TF-IDF pipeline
    - Word2Vec model

- Performs prediction on the test dataset

- Generates:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion matrices
    - Sample false positives and false negatives

The notebook reproduces the final evaluation results reported in the coursework without requiring model retraining.


- Important Notes:


- Ensure all folders remain in the same directory structure to avoid file path errors.
- The notebooks contain Google Drive paths. If running locally, update the paths accordingly.
- NLTK resources required during execution will automatically download.
- Early stopping was applied during LSTM training to reduce overfitting.
- The tokenizer saved in models/tokenizer.pkl must be reused during testing to ensure consistent preprocessing.
- The maximum sequence length used for LSTM padding is 50.
- Word2Vec embeddings were trained directly on the project training dataset using Gensim.


- Reproducibility:

All preprocessing, tokenization, vectorization, and model configurations used during training are included in the notebooks to ensure reproducibility.

The saved preprocessing components and trained models are reused during test evaluation so that predictions are generated using the exact same pipeline applied during training.
