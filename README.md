# Topic-Classification

# I.  Introduction:

#### Project Overview:
The initial development phase of the code base involves implementing a model for Emotion classification based on given dataset:

a) Baseline Emotion Prediction Model with Classic Classifiers:

  - Data Preprocessing: Cleaning and preprocessing data from provided datasets.
  - Feature Engineering: Extracting relevant features for classic classifiers.
  - Model Selection: Choosing classic classifiers like Logistic Regression, Naive Bayes, etc.
  - Model Training: Training classifiers and evaluating performance metrics.
    
b) Generation using pretrained LLMs based on the predicted tag of the model:

  - Prediction Model Integration: After training the baseline topic prediction model using classic classifiers, integrate it with the LLM-based response generation module.

  - Pretrained LLMs Selection: Choose suitable pretrained Large Language Models (LLMs) such as GPT, BERT, or XLNet for response generation. These models possess strong language understanding capabilities and can generate coherent and contextually relevant text.

  - Tag-to-Text Mapping: Develop a mapping mechanism to link predicted topic tags from the baseline model to corresponding text prompts for the LLM. Each topic tag predicted by the baseline model will be associated with a set of text prompts or templates tailored to that specific topic.

  - Contextual Prompt Generation: Generate contextual prompts for the LLM based on the predicted topic tag and any additional context from the user input. These prompts provide relevant information to guide the LLM in generating coherent responses aligned with the predicted topic.

  - Response Generation: Utilize the pretrained LLM to generate responses based on the provided prompts and the context of the conversation. The LLM generates text sequences that are grammatically correct and contextually coherent, incorporating the predicted topic as a guiding factor.

  - UI Chatbot Development: Create a user interface (UI) for the chatbot using web technologies like HTML, CSS, and JavaScript. The UI will allow users to interact with the chatbot seamlessly.

  - Flask API Integration: Develop Flask APIs to connect the UI chatbot with the backend system. These APIs handle user requests, process them through the topic prediction and response generation modules, and return the generated responses to the UI.
  - 

#### Dataset:
Dataset with the content and title of articles most of them in Greek language splitted in two csv's files. 

- TrainData.csv : This file includes the training data for our models

- UnseenData.csv: This file contains the unseed data for the evaluation part.


### Installation


    # go to your home dir
    git clone https://github.com/jvario/topic-classification.git
    cd topic-classification

    # build the image
    docker build -t my-flask-api .
    docker run -d -p 5000:5000 my-flask-api

  #### api - calls
 - **/topic_clf/LoadTopicModel?model_name=SGD** :  loading a model and evaluating it
 - **/topic_clf/TrainTopicModel** : training topic models
# II.  Pipeline:

#### EDA:
After conducting exploratory data analysis (EDA), we noticed that the majority of the content and title columns consist of Greek text with some English phrases interspersed. To address this, we decided to implement text sanitization specifically for the Greek language. This included processes such as removing stopwords and tokenization, which were applied exclusively to the Greek text. Additionally, we utilized a hybrid approach for lemmatization, employing both Greek and English lemmatizers. This ensured that the lemmatization process was effective across both languages present in the dataset.
#### Preproccess:
In order to perform text sanitization on our data, we applied the following steps:

- Joining
- Lowercase
- Clean/Remove NaN values
- Remove HTML tags
- Remove Panctuation
- Remove StopWords
- Tokenization
- Lemmatization

#### Baseline Topic Prediction Model with Classic Classifiers:
In our single-label classification problem, where each data point is associated with only one tag, we carefully divided the dataset into training and testing sets. This division is crucial for evaluating the performance of our models effectively. By separating the data into distinct training and testing subsets, we ensure that our models are trained on one portion of the data and evaluated on another, allowing us to gauge their generalization capability accurately. 

Additionally, for feature extraction, we've applied **TF-IDF (Term Frequency-Inverse Document Frequency)**. This technique helps to represent each document in the dataset as a vector based on the importance of each word, considering both its frequency in the document and its rarity across all documents.

Furthermore, we've utilized **Label Binarization** to encode the single labels into binary format, facilitating the classification task.
For evaluating the performance of our models, we've chosen several metrics including **recall, F1 score, support, Jaccard score, and precision**. These metrics provide insights into different aspects of the model's performance, such as its ability to correctly classify each label, handle imbalanced data, and capture the trade-off between precision and recall.


**Question**:
Suppose there is also the requirement of minimizing the errors of predicting a
sensitive topic article as non-sensitive to zero. Would this introduce any
changes to your modeling approach? Describe any changes you had to make.

**Answer**:
 - **Threshold Adjustment**: One approach is to adjust the classification threshold of the model. By setting a more stringent threshold, the model becomes less likely to classify sensitive articles as non-sensitive. This adjustment ensures that sensitive articles are more accurately identified, reducing the risk of misclassification.
 - **Data Synthesis**: Another strategy involves synthesizing additional data points related to sensitive topics. This could involve augmenting the existing dataset with similar articles or generating synthetic data points that mimic the characteristics of sensitive articles. By providing the model with more examples of sensitive topics, it becomes better equipped to accurately classify such articles.

# III.  Results:

| Model | Sample Size | Accuracy |
|-------|-------------|----------|
| SVM   | ~2500       | 0.88    | 
| SGD   | ~2500       | 0.88     |

This table represents the evaluation results for different models based on dataset. The metrics include Accuracy, Jaccard score.

# IV. Conclusion:
Both the SVM and SGD models have similar accuracy scores of 0.68, indicating that they correctly predicted around 68% of the samples. This level of accuracy is quite commendable, considering the time limitation and dataset constraints. Additionally, when considering the Jaccard Score, which measures the similarity between two sets, the SGD model slightly outperforms the SVM model with a score of 0.56 compared to 0.55.

In conclusion, both models demonstrate relatively good performance within the given limitations, with the SGD model showing a slightly better Jaccard Score. Further analysis, including examining additional performance metrics and conducting cross-validation, would provide a more comprehensive evaluation of the models' effectiveness.
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
