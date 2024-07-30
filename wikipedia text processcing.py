##################################################
# Wikipeadia Text Preprocessing and Visualization
##################################################

from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = (pd.read_csv("wiki_data.csv"))
df.head()

###############################
# Normalizing Case Folding
###############################

def clean_text(dataframe):
    dataframe['text'] = dataframe['text'].str.lower()
    dataframe['text'] = dataframe['text'].str.replace('[w\s]',' ')
    dataframe['text'] = dataframe['text'].str.replace('\d',' ')
    return dataframe

clean_text(df)

df.drop('Unnamed: 0',inplace=True,axis=1)

df.head()

df.reset_index()

df.head()

sw = stopwords.words('english')
def remove_stopwords(dataframe):
    dataframe['text'] = dataframe['text'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    return dataframe

remove_stopwords(df)

###############################
# Rarewords
###############################

drops = pd.Series(' '.join(df['text']).split()).value_counts()[-1000:]

df['text'] = df['text'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))


###############################
# Tokenization
###############################

#nltk.download("punkt")

df["text"].apply(lambda x: TextBlob(x).words).head()


###############################
# Lemmatization
###############################

#nltk.download('wordnet')
df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

###############################
# Barplot
###############################

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

text = " ".join(i for i in df.reviewText)

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()