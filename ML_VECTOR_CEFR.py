from ast import increment_lineno
import pandas as pd
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from nltk.corpus import stopwords
from sklearn import metrics
import csv
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


my_file = '/Users/paulyg/Desktop/Virtro/Python_workspace/Ideas/CEFR_label_vector_tfidf.csv'

df = pd.read_csv(my_file, engine='python')

df.cefr[df.cefr == 'A1'] = 0
df.cefr[df.cefr == 'A2'] = 1
df.cefr[df.cefr == 'B1'] = 2
df.cefr[df.cefr == 'B2'] = 3
df.cefr[df.cefr == 'C1'] = 4
df.cefr[df.cefr == 'C2'] = 4 # Can group the C levels together since they are very proficient to get higher accuracy.

Y = df['cefr'].values
Y=Y.astype('int')

for text in df['text']:
    text = decontracted(text)


X = df['text']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.20, random_state=42)


pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', RandomForestClassifier())
])

model = pipeline.fit(X_train, Y_train)

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix

prediction = model.predict(X_test)

print("accuracy: {}%".format(round(accuracy_score(Y_test, prediction)*100,2)))

rf_cm = metrics.confusion_matrix(Y_test, prediction)
print(rf_cm)

text_to_analyze = StringIO("""Am I applied for this role because I'm really interested in front end like software engineering, I want to. Learn more about how this industry works, and. Learn what I and put what I learned from in the past and what I learn in a classroom and put it in a workplace setting. One of my greatest accomplishments is mate is working on a project that I worked with a group of team members and in that, and with that project we were too. Make a 2D maze game that came out of it and in that meeting we had implementations such as. Like attack, movement and things like that. The best work, not really sure how with the manager was. When I worked with a manager at my old job, we were really respectful for each other. Abuse and respect comes along way, but in a mandatory Alana person like us. So in that in that sense. Working with the major baby learned a lot and had, and we had respect for him and since we had respect for him, we respect was given back to us. I like to take criticism because you always want to learn. About what you did well, you could always do better, since no one's ever perfect, so you can always learn more about things that you don't know about. And since you could take, and since people are giving you criticism, you can take that criticism and see how you can. Work with the feedback that other people are giving towards you. I'm I'm able to delegate. I like I like trusting other team members of. Of tasks that they can handle. And since I'm giving them that trust of task that they can handle. Then I'll be able to work on stuff. That is, give it to me, or if they need help, that I can always help them, or if I ever need help, I can trust it. I can ask them for help as well. A situation when at their work with a client that is different than me. Is when a client and I need wanted two different things, such as they wanted to do this, but I but I thought we should do with this. So what we did was we met in the middle seeing what what I can I can take from his idea and what he could take care of ideas. And since we work from the middle we were to create something that was able to work. For both of our benefits. Well, I was able to get them to sit down and just explain to them what's happening. I know they're frustrated because of doing this project, but. Sit, but even though they're frustrated, 'cause if he's fresh, if they're frustrated, I would also be frustrated about it. I would, I would describe them saying, OK, we apologize for being. I apologize for being frustrated. I apologize for you to be frustrated us. But we can solve this problem by doing this and that and giving him giving him ideas about. What we can do? I would like I like to research. But the software and see how it would be. And by researching it, I'd like to look at small things like look at the big language that we can lose and learn the language. And once I learned that language. I would be able to implement. Functions and the things that are needed for this big project. No. Ah yes, I was faced with a software issue that can be resolved where I was trying to figure. Uh myself where out? Seeing that what I can do and it was and I couldn't really figure out how. Why it was working and I was spending like hours trying to learn and try to figure out what's wrong with it, but it was, but it was hard to figure out, but I learned how. To solve it by lake. Going on Google search up like what? Can I do the work with and stuff like that? I'm currently working on a COBRA 19 tracker. At which tracks which I tracked cases that that you can submit onto A4 manual track AKC the time and date and that put a marker on the Google Maps where it would be at. And I've never had the ability to do that sort of stuff.""")


df2 = pd.read_csv(text_to_analyze, sep=";")

prediction2 = model.predict(df2)
print(prediction2)
#print(X_test)