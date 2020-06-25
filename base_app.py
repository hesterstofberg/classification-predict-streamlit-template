"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# for displaying images
from PIL import Image
wordcloud = Image.open('joint_cloud.png')
wordcount = Image.open('wordcount_bar.png')

# Data dependencies
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words

# Add colors
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# functions for cleaning message
def clean_tweets(message):
	""" We need a docstring here
	"""
    
	#change all words into lower case
	message = message.lower()
    
	#replace website links
	url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'
	web = 'url-web'
	message = re.sub(url, web, message)
    
	#removing puntuation and digits
	message  = "".join([char for char in message if char not in string.punctuation])
	message = re.sub('[0-9]+', '', message)
    
	#removing stopwords
	nltk_stopword = nltk.corpus.stopwords.words('english')
	message = ' '.join([item for item in message.split() if item not in nltk_stopword])
    
	return message

def cleaning (text):
	""" We need a docstring here
	"""
    
	text = re.sub(r'[^\w\s]','',text, re.UNICODE)
	text = text.lower()

	lemmatizer = WordNetLemmatizer()
	text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
	text = [lemmatizer.lemmatize(token, "v") for token in text]

	#text = [word for word in text if not word in stop_words]
	text = " ".join(text)
	text = re.sub('ãââ', '', text)
    
	return text

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Background", "EDA", "Prediction", "Information"]
	st.sidebar.subheader("Navigation")
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Raw data" page
	if selection == "Information":
		st.markdown("Data used for training the model")
		# You can read a markdown file from supporting resources folder
		st.markdown("Here you will find the raw data that was used to train the model, in order to make some predictions.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
            
	# Building out the 'Background' page
	if selection == "Background":
		st.markdown("How it works")
		st.markdown("This web app requires the user to input text (ideally a tweet relating to climate change), and will classify it according to whether or not they believe in climate change. You can have a look at word clouds and other general exloratory data analysis on the 'EDA' page, and make your predictions on the 'Prediction' page that you can navigate to in the sidebar. In the 'Information' page you will find information about the data source and a brief data description.")
		st.markdown("Maybe add some info on how model performance is evaluated? f1 scores and such")
        
	# Building out the 'EDA' page   
	if selection == "EDA":
		st.markdown("Exploratory Data Analysis")   
		st.markdown("Analysis of the training data is an important step in understanding the data. A variety of analysis has been done on the training data. Select an option for more information.")
		if st.checkbox('Word count analysis'):
			st.markdown("Below you will see the wordclouds for each sentiment.")
			# here we can add graphs and word clouds and such        
			st.image(wordcloud, caption='Wordcloud from the training data.', use_column_width=True)
		if st.checkbox('Word frequency analysis'):
			st.markdown("add description")
			st.image(wordcount, caption='Top 20 most frequently used words.', use_column_width=True)
		if st.checkbox('Hashtags for each sentiment'):
			st.markdown("add description")
		if st.checkbox('Average length of each sentiment'):
			st.markdown("add description")

	# Building out the predication page
	if selection == "Prediction":
		st.markdown("Prediction with machine learning models")
		st.markdown("A machine learning model is used to classify tweets about climate change according to three categories. The categories are described below.")
		st.table(pd.DataFrame({'Category': [-1, 0, 1, 2],'Description': ['Anti: this tweet does not believe in man-made climate change', 'Neutral: this tweet neither supports nor refutes the belief of man-made climate change', 'Pro: this tweet supports the belief of man-made climate change', 'News: this tweet links to factual news about climate change']}))
		st.markdown("Enter your opinion on climate change below, then choose what model you would like to use to classify your opinion.")
		# Creating a text box for user input
		tweet_text = st.text_area("What's your opinion on climate change?",'')

        # cleaning text
		tweet_text = clean_tweets(tweet_text)
		tweet_text = cleaning(tweet_text)
		tweet_text = [tweet_text]
        
        # give model choice
		modelChoice = st.radio("Choose a model", ("Linear SVC", "Logistic", "Naive Bayes"))         

		if modelChoice == 'Linear SVC':
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/LN_SVC_model.pkl"),"rb"))
			prediction = predictor.predict(tweet_text)
            
			# When model has successfully run, will print prediction
			st.success("Your opinion has been categorized by the model as: {}".format(prediction))
			st.markdown("This model had the best prediction blah blah blah, maybe add the f1 scores?")
            
		if modelChoice == 'Logistic':
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/LR_model.pkl"),"rb"))
			prediction = predictor.predict(tweet_text)
            
			# When model has successfully run, will print prediction
			st.success("Your opinion has been categorized by the model as: {}".format(prediction))
			st.markdown("This model had the best prediction blah blah blah, maybe add the f1 scores?")        
            
		if modelChoice == 'Naive Bayes':
			# Transforming user input with vectorizer
			#vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/NB_model.pkl"),"rb"))
			prediction = predictor.predict(tweet_text)
            
			# When model has successfully run, will print prediction
			st.success("Your opinion has been categorized by the model as: {}".format(prediction))
			st.markdown("This model had the best prediction blah blah blah, maybe add the f1 scores?")            

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
