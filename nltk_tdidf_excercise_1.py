#!/usr/bin/env python
# coding: utf-8

# ## This is an excercise to find the most frequent words from CountVectorizer and TfidfTransformer
# 
# #### This could be a form of feature engineering if needed

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.express as px
import cufflinks as cf

init_notebook_mode(connected=True)
cf.go_offline()


# In[3]:


import nltk # Imports the library
nltk.download() #Download the necessary datasets


# ### I am using the slogan dataset from Kaggle

# In[4]:


df = pd.read_csv('sloganlist.csv')


# In[5]:


df.head(10)


# In[6]:


df.describe()


# ---

# ### From describe() looks like the csv file has many duplicate data. So I'll just remove all the duplicates

# In[7]:


df = df.drop_duplicates() 
df


# In[8]:


df.describe()


# In[9]:


df[df['Slogan'] == 'Exquisite wodka.']


# ### Two different brands of vodka with the same slogan so I'll let it be

# ---

# ### Let's process the slogans

# In[10]:


# Abbreviations of known words
abbrv_list = ['lol', 'lmao', 'rofl', 'ive', 'youve', 'brb', 'ttyl', 'im']
special_chars_list = ['â', '', '', 'Ã', '©']


# In[11]:


import string
from nltk.corpus import stopwords, wordnet


def text_process(msg):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation and special chars
    2. Replace abbrv with words
    3. Remove all stopwords
    4. Lemmatize words by removing plurals
    5. Returns a list of the cleaned text
    """
    no_punc = []
    for char in msg:
        if char not in string.punctuation and char not in special_chars_list:
            no_punc.append(char)
       
    no_punc = ''.join(no_punc)
    no_punc_word_list = no_punc.split()
    
    cleaned_msg = []
    for word in no_punc_word_list:
        
        if word.lower() not in stopwords.words('english') and word.lower() not in abbrv_list:
            word_lower_case = word.lower()
            word_lemmatized = wordnet.morphy(word_lower_case)
            
            if word_lemmatized is None:
                use_word = word_lower_case
            else:
                use_word = word_lemmatized
                
            cleaned_msg.append(use_word)
    
    cleaned_msg = ' '.join(cleaned_msg)

    return cleaned_msg


# In[12]:


col = 'Slogan'


# In[16]:


cleaned_df = df
cleaned_df[col] = df[col].apply(text_process)


# ---

# ### Here are my cleaned slogans

# In[17]:


cleaned_df[col]


# ---

# ### Vectorization of Slogans

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer


# In[19]:


# tokenize and build vocab
count_vectorizer = CountVectorizer().fit(cleaned_df[col])


# In[20]:


# summarize
print(count_vectorizer.vocabulary_)


# In[21]:


# Vector representation of all msgs
all_msgs_vector = count_vectorizer.transform(cleaned_df[col])
print(all_msgs_vector)


# ---

# ### Applying TF-IDF algorithm

# In[22]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[23]:


tfidf_transformer = TfidfTransformer().fit(all_msgs_vector)


# In[24]:


messages_tfidf = tfidf_transformer.transform(all_msgs_vector)
print(messages_tfidf)


# In[25]:


print(messages_tfidf.shape)


# #### Going by the IDF score should be the best bet since lower score will mean more frequency. 
# #### I will turn the IDF score list into a hasmap and then use the keys of sorted values to find the words 

# In[26]:


idf_score_dict = {}

for i,val in enumerate(tfidf_transformer.idf_):
    idf_score_dict[i] = val   


# In[27]:


import operator
sorted_idf_score_dict = dict(sorted(idf_score_dict.items(), key=operator.itemgetter(1)))


# In[28]:


top_10_idf_score_list = list(sorted_idf_score_dict.keys())[0:10]
top_10_idf_score_list


# In[29]:


unique_word_list = count_vectorizer.get_feature_names()

for i in top_10_idf_score_list:
    try:
        print(unique_word_list[i], idf_score_dict[i])
    except Exception as ex:
        print("no word matching the index")


# ### From the result, it looks like the dataset has a lot of slogans for edible items

# ### Verifying my results for some of these words

# In[30]:


cleaned_df[cleaned_df['Slogan'].str.contains("good")]


# In[31]:


cleaned_df[cleaned_df['Slogan'].str.contains("taste")]


# In[32]:


cleaned_df[cleaned_df['Slogan'].str.contains("food")]


# ### These words are very frequent in the slogans so my results are correct
