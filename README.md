# MedTranscriptClassifier

This project explores the application of natural language processing (NLP) techniques for classifying electronic health records (EHRs) based on medical specialty. Using a dataset of medical transcriptions from various specialties, the project investigates the impact of different preprocessing methods and feature engineering strategies on model performance. The experiments involve varying feature sets, text normalization techniques, and applying domain-specific word embeddings, utilizing machine learning models such as Naïve Bayes, Support Vector Machine (SVM), and fine-tuned Transformer models.

## Project Overview
This project leverages the mtsamples.com dataset, which contains medical transcription samples across multiple specialties and document types. The aim is to classify these transcripts by medical specialty using various NLP techniques and machine learning models. The project is part of the Text Mining in Data Science subject within the Master's program in Data Science at ISCTE – Instituto Universitário de Lisboa.

## Repository Name Suggestions
- MedTranscriptClassifier
- HealthRecordNLP
- MedicalNLPClassify
- EHRSpecialtyClassifier
- MedTextClassification

## Introduction
The present work was conducted within the scope of Text Mining in the Data Science subject, a part of the curriculum for the second semester of the Master's Degree in Data Science at ISCTE – Instituto Universitário de Lisboa. The project aims to classify medical transcripts by specialty using NLP techniques and machine learning models.

## Dataset
The dataset, sourced from mtsamples.com, includes a variety of medical transcription samples across different specialties such as Cardiology, Neurology, Orthopedic, and General Medicine. The dataset comprises 3,792 medical transcription documents with a vocabulary of 40,153 unique words and a total of 142,121 sentences.

## Exploratory Data Analysis
### Corpus
- The dataset contains a variety of medical transcription samples organized by specialty.
- The average word count per document is 459.
- Figure 1 shows the document length distribution.

### Target Variable
- The dataset includes 16 distinct medical specialties after excluding non-specific entries.
- The distribution of records varies significantly across specialties.

## Methods and Results
### Document Pre-processing
- **Handling Missing Values and Class Imbalance:** Removed 31 incomplete records and balanced the dataset by undersampling the most frequent classes and excluding classes with fewer than 50 samples.
- **Text Normalization:** Removed non-alphanumeric characters, digits, and stopwords, and converted text to lowercase.
- **Lemmatization and Stemming:** Compared both techniques using WordNetLemmatizer and PorterStemmer.
- **Tokenization:** Implemented tokenization using the NLTK library.

### Document Processing
- **CountVectorizer:** Transformed text into a matrix of token counts, considering unigrams and excluding terms appearing in more than 80% of documents.
- **TfidfVectorizer:** Applied TF-IDF values with a maximum document frequency of 60% and L2 normalization.
- **Word2Vec:** Trained a Word2Vec model on the corpus and used pre-trained embeddings from glove-wiki-gigaword-100.
- **sBERT:** Used Sentence-BERT to obtain semantically meaningful sentence embeddings.

### Classification Modeling
- **Multinomial Naive Bayes:** Implemented for sparse matrices obtained from CountVectorizer and TfidfVectorizer.
- **Gaussian Naive Bayes:** Applied to dense matrices from Word2Vec embeddings.
- **Support Vector Machine (SVM):** Used with an RBF kernel and tuned regularization parameter.
- **Fine-tuned Transformer Model:** Fine-tuned a pre-trained DistilBERT model for text classification.

## Evaluation
- Models were evaluated using precision, recall, F1-score, and accuracy.
- Weighted averages of these metrics were calculated to account for label imbalance.
- Confusion matrices were used to visualize model performance across all categories.

## Conclusion
The project demonstrated the effectiveness of various NLP techniques and machine learning models for classifying medical transcripts by specialty. While traditional models like Naïve Bayes and SVM provided solid baselines, fine-tuned Transformer models showed promise in handling the complexities of medical text data.

## References
- **Electronic Health Records:** Hartman, R. I., & Lin, J. Y. (2019). Cutaneous melanoma—A review in detection, staging, and management.
- **Text Classification Techniques:** Duarte AF, et al. (2021). Clinical ABCDE rule for early melanoma detection.
- **Machine Learning Models:** Viknesh CK, et al. (2023). Detection and Classification of Melanoma Skin Cancer Using Image Processing Technique.
- **Pre-trained Models:** Codella, N. C. F., et al. (2018). Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging.
- **Word2Vec:** Yuan, Y., & Lo, Y. C. (2017). Improving dermoscopic image segmentation with enhanced convolutional-deconvolutional networks.
- **sBERT:** Sandler, M., et al. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks.
