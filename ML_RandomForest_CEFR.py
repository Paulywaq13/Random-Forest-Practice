from ast import increment_lineno
from io import StringIO
import pandas as pd
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
from nltk.corpus import stopwords
from sklearn import metrics
import syllables
import re
import numpy as np
from random  import shuffle
from tqdm import tqdm

def decontracted(text):
    """
    
    """
    # specific
    text = re.sub(r"won\'t", "will not", text)
    text = re.sub(r"can\'t", "can not", text)

    # general
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    
    return text

def flesch_kinacid_grade_level(string_passage):
    """
    
    """
    string_passage = decontracted(string_passage)
    
    list_words = string_passage.lower().split()

    num_words = len(list_words)

    list_sentences = string_passage.split('.')

    num_sentences = len(list_sentences)
    
    num_syllables = 0

    for words in list_words:
        num_syllables += syllables.estimate(words)
        
    flesch_kinacid_grade = 0.39*(num_words/num_sentences)+11.8*(num_syllables/num_words) - 15.59

    return flesch_kinacid_grade


def flesch_kincaid_score(string_passage):
    """
    
    """
    string_passage = decontracted(string_passage)
    
    list_words = string_passage.lower().split()

    num_words = len(list_words)

    list_sentences = string_passage.split('.')

    num_sentences = len(list_sentences)
    
    num_syllables = 0

    for words in list_words:
        num_syllables += syllables.estimate(words)
        
    #print(num_syllables, num_sentences, num_words)

    flesch_score = 206.835 - 1.015*(num_words/num_sentences) - 84.6*(num_syllables/num_words)
    """
    if flesch_score <= 30 and flesch_score > 0:
        print("Stylistic level: Very Difficult \nGrade Level: University+")
    elif flesch_score > 30 and flesch_score <= 50:
        print("Stylistic level: Difficult \nGrade Level: University/College")
    elif flesch_score > 50 and flesch_score <= 60:
        print("Stylistic level: Fairly Difficult \nGrade Level: High School")
    elif flesch_score > 60 and flesch_score <=70:
        print("Stylistic level: Standard \nGrade Level: 7th-8th grade")
    elif flesch_score > 70 and flesch_score <= 80:
        print("Stylistic level: Fairly Easy \nGrade Level: 6th grade")
    elif flesch_score > 80 and flesch_score <= 90:
        print("Stylistic level: Easy \nGrade Level: 5th Grade")
    """ 
    
    #print('Flesch Score: ', end="")
    return int(flesch_score)


def ma_ttr_lex_diversity(text):
    """
    ma_ttr stands for moving average type-token ratio. Similar to the type-token ratio used to analyze
    lexical diversity of a text, ma_ttr analyzes in 50 word, overlapping increments. Example is taking the 1-50th group of words
    then the 2-51st, etc. An average is then taken to find a more accurate type-token ratio and thus better analysis of text.

    Args:
        text (str): string of text document to be analyzed.
        Returns: (int) ma_ttr score.
    """
    
    window_length = 50
    
    text = decontracted(text)
    
    text = re.sub(r'[^\w]', ' ', text)
    
    text = text.lower().split()
    
    if len(text) < (window_length + 1):
        ma_ttr = len(set(text))/len(text)
       
    else:
        sum_ttr = 0
        denom = 0
        
        for x in range(len(text)):
            
            segment = text[x:(x + window_length)]
        
            if len(segment) < window_length:
                break
            
            denom += 1
            sum_ttr += (len(set(segment))/float(window_length))
            
        ma_ttr = (sum_ttr/denom) * 100
    
    return(ma_ttr)


my_file = '/Users/paulyg/Desktop/Virtro/Python_workspace/Ideas/CEFR_label_complete3.csv'

my_file2 = '/Users/paulyg/Desktop/Virtro/Python_workspace/Ideas/CEFR_READABILITY_ALL.csv'

french_file = '/Users/paulyg/Desktop/Desktop/Virtro/Python_workspace/Ideas/CEFR_label_FRENCH2.csv'

def random_forest_cefr(filename):

    df = pd.read_csv(filename, engine='python')

    #print(df.head(10))

    #print(df.info())
    df.cefr[df.cefr == 'A1'] = 0
    df.cefr[df.cefr == 'A2'] = 1
    df.cefr[df.cefr == 'B1'] = 2
    df.cefr[df.cefr == 'B2'] = 3
    df.cefr[df.cefr == 'C1'] = 4
    df.cefr[df.cefr == 'C2'] = 4 # Can group the C levels together since they are very proficient to get higher accuracy.

    #print(df.head(100))

    #Define dependent variable
    Y = df['cefr'].values
    Y=Y.astype('int')

    #Define independent variables, this could be readaiblity, MA_TTR, etc, etc

    X = df.drop(labels=['cefr'], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=42)

    model = RandomForestClassifier(n_estimators = 100, random_state=42)

    model.fit(X_train, Y_train)

    prediction_test = model.predict(X_test)

    print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))

    feature_list = list(X.columns)

    feature_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)

    print(feature_imp)
    print(X_test)
    text_to_analyze = """I'm a bit concerned about some of the divisions I see within Hellenic polytheism these days.  It's not the divisions themselves which concern me, because variation is good. We shouldn't be overspecialized, but should represent the diversity present within the community.  Hellenic polytheism includes, but isn't limited to Hellenion and other 'Hellenic Reconstructionist' groups and individuals, Hellenic wiccans, people devoted to Hellenic deities but not all that concerned with reconstructionism, a 'group' (for lack of a better word) called 'Hellenic Traditionalists' and more.  What  does  concern me is what seems like divisive behavior driving potential wedges between us all in ways that aren't necessarily healthy or helpful. I'm talking about antagonistic behavior, of which any of us can be guilty from time to time, but which I've been witnessing more than I feel comfortable with here lately.  Drew Campbell's book is not beyond criticism, but it's also far from worthless. I've seen Hellenion called too liberal, too conservative, and that it tries to be all things to all people, and the simple fact is that these can not all be objectively true at the same time. People make statements reflecting their subjective impressions, and those statements are objectified (sometimes by the speaker, and sometimes by offended readers/listeners).  Can't we all try to get along?  Since the  very first day  that sponde.com came into being, I've tried to get people from various camps to contribute articles, hymns, and pretty much  anything  else. But so far, almost every word of content on the site has come from members of Hellenion, the organization that everyone seems to love to hate. Does this mean that only Hellenion members are publishing things of value? I'd hesitate to go that far, but I would much rather be able to offer a definite  NO .  I'm not sure I have one simple solution to this problem, but I do think that we'd be better off if we poured more of this energy into actual Hellenic polytheism. Isn't that something we can agree on enough to put all this other stuff to bed?"""

    flesch_grade = flesch_kinacid_grade_level(text_to_analyze)
    flesch_score = flesch_kincaid_score(text_to_analyze)
    ma_ttr = ma_ttr_lex_diversity(text_to_analyze)
    
    text42 = StringIO(f"flesch_grade; flesch_score; ma_ttr\n{flesch_grade};{flesch_score};{ma_ttr}")
    
    df2 = pd.read_csv(text42, sep=";")
    prediction2 = model.predict(df2)
    print(prediction2)
    
random_forest_cefr(my_file)


