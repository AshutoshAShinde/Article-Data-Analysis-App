# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:25:50 2021

@author: Lenovo
"""
import streamlit as st
from wordcloud import WordCloud
import pandas as pd
import matplotlib.image as mpimg
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')

import matplotlib.pyplot as plt
import seaborn as sns
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
pd.set_option('display.max_colwidth', -1)

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Article Data Analysis App')
images = mpimg.imread('text3.jpg')
st.image(images, height = 600, width = 600)

st.sidebar.title("Navigation")   
category = ["Article Data Analysis App","Source Code"]
choice = st.sidebar.radio("Select the Navigation", category)

st.sidebar.title("Created By:")
st.sidebar.subheader("Ashutosh Shinde")
st.sidebar.subheader("[LinkedIn Profile](https://www.linkedin.com/in/ashutoshashinde/)")
st.sidebar.subheader("[GitHub Repository](https://github.com/AshutoshAShinde/Article-Data-Analysis-App)")
if choice == "Article Data Analysis App":

    article = st.text_area("Enter the Article that you want to perform the Analysis on")
    st.subheader("Paste the Article in the text area above and hit Ctrl + Enter ")
    
    st.subheader("This App performs the Sentiment Analysis for any Article")
    
    st.graphviz_chart("""
        digraph{
        Article -> Sentiment
        Sentiment -> Positive
        Sentiment -> Neutral 
        Sentiment -> Negative
        Positive -> MostPositiveSentence
        Positive -> WordCloud
        Positive -> WordFrequency
        Negative -> MostNegativeSentence
        Negative -> WordCloud
        Negative -> WordFrequency
        Neutral -> WordCloud
        Neutral -> WordFrequency
        }
        """)
    st.write("Note: The error goes away once you paste the Article")
    st.subheader("The functions performed by the Analyzer are :")
    sent = nltk.tokenize.sent_tokenize(article)
    dfs = pd.DataFrame(sent,columns=['Sentence'])
    
    senti_analyzer = SentimentIntensityAnalyzer()
    compound_score = [] 

    for sen in dfs['Sentence']:
    
        compound_score.append(senti_analyzer.polarity_scores(sen)['compound'])
    
    dfs['Compound Score'] = compound_score
    
    Sentiment = []

    for i in compound_score:
    
        if i >= 0.05:
        
            Sentiment.append('Positive')
        
        elif i > -0.05 and i < 0.05:
        
            Sentiment.append('Neutral')
        
        else:
        
            Sentiment.append('Negative')
            
    dfs['Sentiment'] = Sentiment
    stopwords = stopwords.words('english')
    pos_count = sum(dfs['Sentiment']=='Positive')
    neg_count = sum(dfs['Sentiment']=='Negative')
    neu_count = sum(dfs['Sentiment']=='Neutral')
    
    labels = ['Positive Sentences', 'Negative Sentences', 'Neutral Sentences']
    sizes = [pos_count, neg_count, neu_count]
    colors = ['#66b3ff','#ff9999', '#c27ba0']
    #explsion
    explode = (0.05,0.05,0.05)
    
    pos_max = dfs.loc[dfs['Compound Score']==max(dfs['Compound Score'])]
    
    neg_max = dfs.loc[dfs['Compound Score']==min(dfs['Compound Score'])]
    
    gp = dfs.groupby(by=['Sentiment'])
    positive_sent = gp.get_group('Positive')  
    negative_sent = gp.get_group('Negative')    
    neutral_sent = gp.get_group('Neutral')
      
    def plot_Cloud(wordCloud):
        plt.figure( figsize=(20,10), facecolor='w')
        plt.imshow(wordCloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
    
    def wordcloud(data):
    
        words_corpus = ''
        words_list = []

    
        for rev in data["Sentence"]:
        
            text = str(rev).lower()
            text = text.replace('rt', ' ') 
            text = re.sub(r"http\S+", "", text)        
            text = re.sub(r'[^\w\s]','',text)
            text = ''.join([i for i in text if not i.isdigit()])
        
            tokens = nltk.word_tokenize(text)
            tokens = [word for word in tokens if word not in stopwords]
        
        # Remove aplha numeric characters
        
            for words in tokens:
            
                words_corpus = words_corpus + words + " "
                words_list.append(words)
            
        return words_corpus, words_list
    
    positive_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(positive_sent)[0])    
    negative_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(negative_sent)[0])
    neutral_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(neutral_sent)[0])    
    total_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(dfs)[0])
    
    at = nltk.FreqDist(wordcloud(dfs)[1])
    dt = pd.DataFrame({'Wordcount': list(at.keys()),
                  'Count': list(at.values())})
    # selecting top 10 most frequent hashtags     
    dt = dt.nlargest(columns="Count", n = 10) 

    ap = nltk.FreqDist(wordcloud(positive_sent)[1])
    dp = pd.DataFrame({'Wordcount': list(ap.keys()),
                  'Count': list(ap.values())})
    # selecting top 10 most frequent hashtags     
    dp = dp.nlargest(columns="Count", n = 10) 

    an = nltk.FreqDist(wordcloud(negative_sent)[1])
    dn = pd.DataFrame({'Wordcount': list(an.keys()),
                  'Count': list(an.values())})
    # selecting top 10 most frequent hashtags     
    dn = dn.nlargest(columns="Count", n = 10) 

    au = nltk.FreqDist(wordcloud(neutral_sent)[1])
    du = pd.DataFrame({'Wordcount': list(au.keys()),
                  'Count': list(au.values())})
    # selecting top 10 most frequent hashtags     
    du = du.nlargest(columns="Count", n = 10) 
    
    st.write("1. Displays the Sentiment Distribution of the Article ")
    st.write("2. Generates the Wordcloud for the Article")
    st.write("3. Displays the Most Frequent Words used in the Article")
    
    Analyzer_choice1 = st.selectbox("Select the Option",  ["Display the Sentiment Distribution of the Article","Generate the Wordcloud for the Article","Display the Most Frequent Words used in the Article"])
    
    if (st.button("Analyze")):
        
        if Analyzer_choice1 == "Display the Sentiment Distribution of the Article":
            
            st.write("Sentiment Distribution of the Article")
            st.write(plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode))
            st.pyplot(use_container_width=True)
            
        elif Analyzer_choice1 == "Generate the Wordcloud for the Article":
            
            st.write("Wordcloud of the Article")
            st.write(plot_Cloud(total_wordcloud))
            st.pyplot(use_container_width=True)
            
        else:
            
            #st.write(neg_max[:1])
            st.write('Most Frequent Words used in the Article')
            st.write(sns.barplot(data=dt, y= "Wordcount", x = "Count"))
            st.pyplot(use_container_width=True)
    
    st.write("4. Displays the Positive Sentences of the Article ")
    st.write("5. Displays the Most Positive Sentence in the Article")
    st.write("6. Generates a WordCloud for the Positive Sentences of the Article")
    st.write("7. Displays the Most Frequent Words used in the Positive Sentences of the Article")
    
    Analyzer_choice2 = st.selectbox("Select the Option",  ["Display the Positive Sentences of the Article","Display the Most Positive Sentence of the Article","Generate a WordCloud for the Positive Sentences of the Article","Display the Most Frequent Words used in the Positive Sentences of the Article"])
     
    if (st.button("Analyze ")):
        
        if Analyzer_choice2 == "Display the Positive Sentences of the Article":
            
            st.write('Positive Sentences in the Article')
            st.table(positive_sent.head())
            
        elif Analyzer_choice2 == "Display the Most Positive Sentence of the Article":
            
            st.write('Most Positive Sentence in the Article')
            st.table(pos_max[:1])
            
        elif Analyzer_choice2 == "Generate a WordCloud for the Positive Sentences of the Article":
            
            #positive_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(positive_tweets)[0])
            st.write('WordCloud of the Positive Sentences of the Article')
            st.write(plot_Cloud(positive_wordcloud))
            st.pyplot(use_container_width=True)

            
        elif Analyzer_choice2 == "Display the Most Frequent Words used in the Positive Sentences of the Article":
            
            st.write('Most Frequent Words in the Positive Sentences')
            st.write(sns.barplot(data=dp,y= "Wordcount", x = "Count"))
            st.pyplot(use_container_width=True)

    st.write("8. Displays the Negative Sentences of the Article ")
    st.write("9. Displays the Most Negative Sentence of the Article")
    st.write("10. Generates a WordCloud for the Negative Sentences of the Article")
    st.write("11. Displays the Most Frequent Words used in the Negative Sentences of the Article")    

    
    Analyzer_choice3 = st.selectbox("Select the Option",  ["Display the Negative Sentences of the Article","Display the Most Negative Sentence of the Article","Generate a WordCloud for the Negative Sentences of the Article", "Display the Most Frequent Words used in the Negative Sentences of the Article"])
                                                            
    if st.button("Analyze  "):
            
        if Analyzer_choice3 == "Display the Negative Sentences of the Article":
            
            st.write('Negative Sentences in the Article')
            st.table(negative_sent.head())            

        elif Analyzer_choice3 == "Display the Most Negative Sentence of the Article":

            st.write('Most Negative Sentence in the Article')
            st.table(neg_max[:1])
            
        elif Analyzer_choice3 == "Generate a WordCloud for the Negative Sentences of the Article":
            
            st.write('WordCloud of the Negative Sentences in the Article')
            #negative_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(negative_tweets)[0])
            st.write(plot_Cloud(negative_wordcloud))
            st.pyplot(use_container_width=True)
            
        else:
            
            st.write('Most Frequent Words in the negative Sentences')
            st.write(sns.barplot(data=dn, y= "Wordcount", x = "Count"))
            st.pyplot(use_container_width=True)
            
    st.write("12. Displays the Neutral Sentences of the Article ")
    st.write("13. Generates a WordCloud for the Neutral Sentences of the Article")
    st.write("14. Displays the Most Frequent Words used in the Neutral Sentences of the Article")
            
    Analyzer_choice4 = st.selectbox("Select the Option",  ["Display the Neutral Sentences of the Article","Generate a WordCloud for the Neutral Sentences of the Article", "Display the Most Frequent Words used in the Neutral Sentences of the Article"])
                                                            
    if (st.button("Analyze   ")):

        if Analyzer_choice4 == "Neutral Sentences in the Article":
        
            st.write('Display the Neutral Sentences of the Article')
            st.table(neutral_sent.head())

        elif Analyzer_choice4 == "Generate a WordCloud for the Neutral Sentences of the Article":
            
            st.write('WordCloud of the Neutral Sentences in the Article')
            st.write(plot_Cloud(neutral_wordcloud))
            st.pyplot(use_container_width=True)
            
        else:
            
            st.write('Most Frequent Words in the Negative Sentences')
            st.write(sns.barplot(data=du, y= "Wordcount", x = "Count"))
            st.pyplot(use_container_width=True)
    
    
else:
    
    st.subheader("Source Code")
    
    code="""

import streamlit as st
from wordcloud import WordCloud
import pandas as pd
import re
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
pd.set_option('display.max_colwidth', -1)
warnings.filterwarnings("ignore")

import nltk
from nltk.corpus import stopwords

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title('Article Data Analysis App')
images = cv2.imread('text3.jpg')
st.image(images, height = 600, width = 600)

st.sidebar.title("Navigation")   
category = ["Article Data Analysis App","Source Code"]
choice = st.sidebar.radio("Select the Navigation", category)

st.sidebar.title("Created By:")
st.sidebar.subheader("Ashutosh Shinde")
st.sidebar.subheader("[LinkedIn Profile](https://www.linkedin.com/in/ashutoshashinde/)")
st.sidebar.subheader("[GitHub Repository](https://github.com/AshutoshAShinde/Article-Data-Analysis-App)")
if choice == "Article Data Analysis App":

    article = st.text_area("Enter the Article that you want to perform the Analysis on")
    st.subheader("Paste the Article in the text area above and hit Ctrl + Enter ")
    
    st.subheader("This App performs the Sentiment Analysis for any Article")
    
    st.graphviz_chart(""
        digraph{
        Article -> Sentiment
        Sentiment -> Positive
        Sentiment -> Neutral 
        Sentiment -> Negative
        Positive -> MostPositiveSentence
        Positive -> WordCloud
        Positive -> WordFrequency
        Negative -> MostNegativeSentence
        Negative -> WordCloud
        Negative -> WordFrequency
        Neutral -> WordCloud
        Neutral -> WordFrequency
        }
        "")
    st.write("Note: The error goes away once you paste the Article")
    st.subheader("The functions performed by the Analyzer are :")
    sent = nltk.tokenize.sent_tokenize(article)
    dfs = pd.DataFrame(sent,columns=['Sentence'])
    
    senti_analyzer = SentimentIntensityAnalyzer()
    compound_score = [] 

    for sen in dfs['Sentence']:
    
        compound_score.append(senti_analyzer.polarity_scores(sen)['compound'])
    
    dfs['Compound Score'] = compound_score
    
    Sentiment = []

    for i in compound_score:
    
        if i >= 0.05:
        
            Sentiment.append('Positive')
        
        elif i > -0.05 and i < 0.05:
        
            Sentiment.append('Neutral')
        
        else:
        
            Sentiment.append('Negative')
            
    dfs['Sentiment'] = Sentiment
    
    pos_count = sum(dfs['Sentiment']=='Positive')
    neg_count = sum(dfs['Sentiment']=='Negative')
    neu_count = sum(dfs['Sentiment']=='Neutral')
    
    labels = ['Positive Sentences', 'Negative Sentences', 'Neutral Sentences']
    sizes = [pos_count, neg_count, neu_count]
    colors = ['#66b3ff','#ff9999', '#c27ba0']
    #explsion
    explode = (0.05,0.05,0.05)
    
    pos_max = dfs.loc[dfs['Compound Score']==max(dfs['Compound Score'])]
    
    neg_max = dfs.loc[dfs['Compound Score']==min(dfs['Compound Score'])]
    
    gp = dfs.groupby(by=['Sentiment'])
    positive_sent = gp.get_group('Positive')  
    negative_sent = gp.get_group('Negative')    
    neutral_sent = gp.get_group('Neutral')
      
    def plot_Cloud(wordCloud):
        plt.figure( figsize=(20,10), facecolor='w')
        plt.imshow(wordCloud)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()
        
    stopwords = nltk.corpus.stopwords.words('english')
    
    def wordcloud(data):
    
        words_corpus = ''
        words_list = []

    
        for rev in data["Sentence"]:
        
            text = str(rev).lower()
            text = text.replace('rt', ' ') 
            text = re.sub(r"http\S+", "", text)        
            text = re.sub(r'[^\w\s]','',text)
            text = ''.join([i for i in text if not i.isdigit()])
        
            tokens = nltk.word_tokenize(text)
            tokens = [word for word in tokens if word not in stopwords]
        
        # Remove aplha numeric characters
        
            for words in tokens:
            
                words_corpus = words_corpus + words + " "
                words_list.append(words)
            
        return words_corpus, words_list
    
    positive_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(positive_sent)[0])    
    negative_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(negative_sent)[0])
    neutral_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(neutral_sent)[0])    
    total_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(dfs)[0])
    
    at = nltk.FreqDist(wordcloud(dfs)[1])
    dt = pd.DataFrame({'Wordcount': list(at.keys()),
                  'Count': list(at.values())})
    # selecting top 10 most frequent hashtags     
    dt = dt.nlargest(columns="Count", n = 10) 

    ap = nltk.FreqDist(wordcloud(positive_sent)[1])
    dp = pd.DataFrame({'Wordcount': list(ap.keys()),
                  'Count': list(ap.values())})
    # selecting top 10 most frequent hashtags     
    dp = dp.nlargest(columns="Count", n = 10) 

    an = nltk.FreqDist(wordcloud(negative_sent)[1])
    dn = pd.DataFrame({'Wordcount': list(an.keys()),
                  'Count': list(an.values())})
    # selecting top 10 most frequent hashtags     
    dn = dn.nlargest(columns="Count", n = 10) 

    au = nltk.FreqDist(wordcloud(neutral_sent)[1])
    du = pd.DataFrame({'Wordcount': list(au.keys()),
                  'Count': list(au.values())})
    # selecting top 10 most frequent hashtags     
    du = du.nlargest(columns="Count", n = 10) 
    
    st.write("1. Displays the Sentiment Distribution of the Article ")
    st.write("2. Generates the Wordcloud for the Article")
    st.write("3. Displays the Most Frequent Words used in the Article")
    
    Analyzer_choice1 = st.selectbox("Select the Option",  ["Display the Sentiment Distribution of the Article","Generate the Wordcloud for the Article","Display the Most Frequent Words used in the Article"])
    
    if (st.button("Analyze")):
        
        if Analyzer_choice1 == "Display the Sentiment Distribution of the Article":
            
            st.write("Sentiment Distribution of the Article")
            st.write(plt.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode))
            st.pyplot(use_container_width=True)
            
        elif Analyzer_choice1 == "Generate the Wordcloud for the Article":
            
            st.write("Wordcloud of the Article")
            st.write(plot_Cloud(total_wordcloud))
            st.pyplot(use_container_width=True)
            
        else:
            
            #st.write(neg_max[:1])
            st.write('Most Frequent Words used in the Article')
            st.write(sns.barplot(data=dt, y= "Wordcount", x = "Count", color="b"))
            st.pyplot(use_container_width=True)
    
    st.write("4. Displays the Positive Sentences of the Article ")
    st.write("5. Displays the Most Positive Sentence in the Article")
    st.write("6. Generates a WordCloud for the Positive Sentences of the Article")
    st.write("7. Displays the Most Frequent Words used in the Positive Sentences of the Article")
    
    Analyzer_choice2 = st.selectbox("Select the Option",  ["Display the Positive Sentences of the Article","Display the Most Positive Sentence of the Article","Generate a WordCloud for the Positive Sentences of the Article","Display the Most Frequent Words used in the Positive Sentences of the Article"])
     
    if (st.button("Analyze ")):
        
        if Analyzer_choice2 == "Display the Positive Sentences of the Article":
            
            st.write('Positive Sentences in the Article')
            st.table(positive_sent.head())
            
        elif Analyzer_choice2 == "Display the Most Positive Sentence of the Article":
            
            st.write('Most Positive Sentence in the Article')
            st.table(pos_max[:1])
            
        elif Analyzer_choice2 == "Generate a WordCloud for the Positive Sentences of the Article":
            
            #positive_wordcloud = WordCloud(background_color= "white",width=900, height=500).generate(wordcloud(positive_tweets)[0])
            st.write('WordCloud of the Positive Sentences of the Article')
            st.write(plot_Cloud(positive_wordcloud))
            st.pyplot(use_container_width=True)

            
        elif Analyzer_choice2 == "Display the Most Frequent Words used in the Positive Sentences of the Article":
            
            st.write('Most Frequent Words in the Positive Sentences')
            st.write(sns.barplot(data=dp,y= "Wordcount", x = "Count", color="b"))
            st.pyplot(use_container_width=True)

    st.write("8. Displays the Negative Sentences of the Article ")
    st.write("9. Displays the Most Negative Sentence of the Article")
    st.write("10. Generates a WordCloud for the Negative Sentences of the Article")
    st.write("11. Displays the Most Frequent Words used in the Negative Sentences of the Article")    

    
    Analyzer_choice3 = st.selectbox("Select the Option",  ["Display the Negative Sentences of the Article","Display the Most Negative Sentence of the Article","Generate a WordCloud for the Negative Sentences of the Article", "Display the Most Frequent Words used in the Negative Sentences of the Article"])
                                                            
    if st.button("Analyze  "):
            
        if Analyzer_choice3 == "Display the Negative Sentences of the Article":
            
            st.write('Negative Sentences in the Article')
            st.table(negative_sent.head())            

        elif Analyzer_choice3 == "Display the Most Negative Sentence of the Article":

            st.write('Most Negative Sentence in the Article')
            st.table(neg_max[:1])
            
        elif Analyzer_choice3 == "Generate a WordCloud for the Negative Sentences of the Article":
            
            st.write('WordCloud of the Negative Sentences in the Article')
            #negative_wordcloud = WordCloud(width=900, height=500).generate(wordcloud(negative_tweets)[0])
            st.write(plot_Cloud(negative_wordcloud))
            st.pyplot(use_container_width=True)
            
        else:
            
            st.write('Most Frequent Words in the negative Sentences')
            st.write(sns.barplot(data=dn, y= "Wordcount", x = "Count", color="b"))
            st.pyplot(use_container_width=True)
            
    st.write("12. Displays the Neutral Sentences of the Article ")
    st.write("13. Generates a WordCloud for the Neutral Sentences of the Article")
    st.write("14. Displays the Most Frequent Words used in the Neutral Sentences of the Article")
            
    Analyzer_choice4 = st.selectbox("Select the Option",  ["Display the Neutral Sentences of the Article","Generate a WordCloud for the Neutral Sentences of the Article", "Display the Most Frequent Words used in the Neutral Sentences of the Article"])
                                                            
    if (st.button("Analyze   ")):

        if Analyzer_choice4 == "Neutral Sentences in the Article":
        
            st.write('Display the Neutral Sentences of the Article')
            st.table(neutral_sent.head())

        elif Analyzer_choice4 == "Generate a WordCloud for the Neutral Sentences of the Article":
            
            st.write('WordCloud of the Neutral Sentences in the Article')
            st.write(plot_Cloud(neutral_wordcloud))
            st.pyplot(use_container_width=True)
            
        else:
            
            st.write('Most Frequent Words in the Negative Sentences')
            st.write(sns.barplot(data=du, y= "Wordcount", x = "Count"), color="b")
            st.pyplot(use_container_width=True)

    
    """
    st.code(code, language='python')

