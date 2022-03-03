import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')


class svm_comp:
    np.random.seed(500)
    # insert path to csv file
    print("Please input the absolute path to the corpus file")
    path = input()
    # "D:\Cam_Y2\Group project\Question classifier\test_csv.csv"
    # "D:\Cam_Y2\Group project\Question classifier\version1.csv"
    dtype={'text': str, 'label': str}
    Corpus = pd.read_csv(path, dtype,encoding='latin-1')

    # Step - a : Remove blank rows if any.
    Corpus['text'].dropna(inplace=True)

    # Step - b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    Corpus['text'] = [entry.lower() for entry in Corpus['text']]
    # (Alex) Website said: Corpus['text'] = [entry.str.lower() for entry in Corpus['text']]

    # Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
    Corpus['text']= [word_tokenize(entry) for entry in Corpus['text']]
    # (Alex) 
    # I'm not really sure what is going on here, but the imported data is already in tocken form. The series Corpus['text'] 
    # contains lists of words, which is essentially the tockenised form. Supposedly, the author of the webpage got a string 
    # as each entry in Corpus['text'], thus he used ".lower()" on the entry, then try to tockenise it.
    # Finally, the above prosedure does not take signifigantly more time, plus this should be done at configure time, so it should be fine

    # Step - d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun

    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index,entry in enumerate(Corpus['text']):
        # Declaring Empty List to store the words that follow the rules for this step
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(entry):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Corpus.loc[index,'text_final'] = str(Final_words)

    Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(Corpus['text_final'],Corpus['label'],test_size=0.3)

    Encoder = LabelEncoder()
    Train_Y = Encoder.fit_transform(Train_Y)
    Test_Y = Encoder.fit_transform(Test_Y)

    Tfidf_vect = TfidfVectorizer(max_features=5000)
    Tfidf_vect.fit(Corpus['text_final'])
    Train_X_Tfidf = Tfidf_vect.transform(Train_X)
    Test_X_Tfidf = Tfidf_vect.transform(Test_X)

    # Classifier - Algorithm - SVM
    # fit the training dataset on the classifier
    SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
    SVM.fit(Train_X_Tfidf,Train_Y)

    # predict the labels on validation dataset
    predictions_SVM = SVM.predict(Test_X_Tfidf)
    # Use accuracy_score function to get the accuracy
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)


    def classify(input, SVM,Tfidf_vect):
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        in_tokens = word_tokenize(input)
        Final_words = []
        # Initializing WordNetLemmatizer()
        word_Lemmatized = WordNetLemmatizer()
        # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
        for word, tag in pos_tag(in_tokens):
            # Below condition is to check for Stop words and consider only alphabets
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
        Test_X_Tfidf = Tfidf_vect.transform([str(Final_words)])

        return SVM.predict(Test_X_Tfidf)


    classify("buddy what is up, wanna hang out?",SVM,Tfidf_vect)
    classify("I think I have cancer, I fell pain, I feel unconfortable, pls help",SVM,Tfidf_vect)
    classify("I don't know man, is this machine working? I sure damn hope so.",SVM,Tfidf_vect)
    inner = ""
    while(inner != "stop"):
        print("please enter a question")
        inner = input()
        response = classify(inner,SVM,Tfidf_vect)
        if(response==0):
            print("medical question")
        else:
            print("casual question")
        print()


