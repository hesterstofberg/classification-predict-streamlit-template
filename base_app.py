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

#images on home page
sentiment_analysis = Image.open('resources/imgs/Sentiment-Analysis.png')
climate = Image.open('resources/imgs/Climate Change.jpg')
globe = Image.open('resources/imgs/globe.png')

# images on how it works page
How_it_works = Image.open('resources/imgs/How.png')
Home_Page = Image.open('resources/imgs/Home.PNG')
How_to_Page = Image.open('resources/imgs/How2.PNG')
EDA_Page = Image.open('resources/imgs/EDA.PNG')
Predictions_Page = Image.open('resources/imgs/Prediction.PNG')
Resources_Page = Image.open('resources/imgs/Resources.PNG')
About_Page = Image.open('resources/imgs/About.PNG')

# images on EDA page
unbalancedData = Image.open('resources/imgs/unbalanced.png')
anti_hashtags = Image.open('resources/imgs/negative_hashtags.png')
neutral_hashtags = Image.open('resources/imgs/neutral_hashtags.png')
news_hashtags = Image.open('resources/imgs/news_hashtags.png')
pro_hashtags = Image.open('resources/imgs/positive_hashtags.png')
wordcloud = Image.open('resources/imgs/joint_cloud(2).png')

# images on predictions page
classification = Image.open('resources/imgs/classification.jpg')
pro_text = Image.open('resources/imgs/positive.png')
anti_text = Image.open('resources/imgs/negative.png')
neutral_text = Image.open('resources/imgs/neutral.png')
news_text = Image.open('resources/imgs/news-emoji.png')

# images on resources page 
twitter = Image.open('resources/imgs/twitter.png')

# images on about us page
Anna = Image.open('resources/imgs/Anna.png')
banner = Image.open('resources/imgs/banner.jpg')
Buhle = Image.open('resources/imgs/Buhle.jpg')
contact = Image.open('resources/imgs/contact.jpg')
Hester = Image.open('resources/imgs/Hester.png')
Maddy = Image.open('resources/imgs/Maddy.png')
Olwethu = Image.open('resources/imgs/Olwethu.png')
Tony = Image.open('resources/imgs/Tony.jpg')

# Data dependencies
import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
#stop_words = set(stopwords.words("english"))
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words

# Add background and text colors
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# Load raw data
raw = pd.read_csv("resources/train.csv")

# Functions for cleaning text
def clean_tweets(message):
	""" This function removes punctions and url's from the message"""
    
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

# Function for cleaning text
def cleaning (text):
	"""this function lemmatizes the message"""
    
	text = re.sub(r'[^\w\s]','',text, re.UNICODE)
	text = text.lower()

	lemmatizer = WordNetLemmatizer()
	text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
	text = [lemmatizer.lemmatize(token, "v") for token in text]

	text = " ".join(text)
	text = re.sub('ãââ', '', text)
    
	return text

# Function to give statement of prediction
def statement(sentiment):
	"""This function gives a statement according to the prediction made"""
	# Statement for 'anti' text
	if sentiment == -1:
		st.image(anti_text, caption='Climate change is not real.', width=250)
		st.success("The selected model has determined that the given text does not believe in man-made climate change")
	# Statement for 'neutral' text      
	if sentiment == 0:
		st.image(neutral_text, caption='Fence sitting.', width=250)
		st.success("The selected model has determined that the given text neither supports nor refutes the belief of man-made climate change")
	# Statement for 'pro' text        
	if sentiment == 1:
		st.image(pro_text, caption='Climate change is real!', width=250)
		st.success("The selected model has determined that the given text supports the belief of man-made climate change")
	# Statement for 'news' text        
	if sentiment == 2:
		st.image(news_text, caption='Factual/news related.', width=250)
		st.success("The selected model has determined that the given text links to factual news about climate change")
        
	return     

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Home","How it works", "EDA", "Making Predictions", "Resources", "About us"]
	st.sidebar.subheader("Navigation")
	selection = st.sidebar.selectbox("Choose a page", options)

	# Building the home page    
	if selection == "Home":
		st.title("Classification Station:tm:")
		st.subheader("Welcome!:grin:")
		st.markdown("This web app aims to classifify text related to climate change.")        
		st.image(sentiment_analysis, caption='', width=500)        
		st.subheader("Why climate change?")
		st.markdown("Many companies are built around lessening one’s environmental impact or carbon footprint. They offer products and services that are environmentally friendly and sustainable, in line with their values and ideals. They would like to determine how people perceive climate change and whether or not they believe it is a real threat. This would add to their market research efforts in gauging how their product/service may be received.")
		st.image(climate, caption='Climate change is in our hands.', use_column_width=True)        
            
	# Building out the 'How it works' page
	if selection == "How it works":
		st.title("How it works")
		st.image(How_it_works, caption='', width=300)        
		st.subheader("Navigation Bar")
		st.markdown("Home Screen  - Landing page, gives brief discription of the App")
		st.image(Home_Page, width=500)
		st.markdown("How it works - Explains how the app works, from navigating on pages, to how efficient the model used is.")
		st.image(How_to_Page, width=500)
		st.markdown("EDA          - Exploratory data analysis shows how we analysing the tweets data sets to summarize their main characteristics, using visuals. EDA is basically for displaying what the data can tell us beyond the formal modeling.")
		st.image(EDA_Page, width=500)
		st.markdown("Making Predictions - This is where the magic happens. This is the part where a tweet can be typed in and be classified based on the models we created. User aslo has option on which model they want to use for classification. This page also shows the accuracy score of the model used. This is a reflection of efficiency of the used model.")
		st.image(Predictions_Page, width=500)
		st.markdown("Resources - This page show the raw tweeter data used for training the model.")
		st.image(Resources_Page, width=500)
		st.markdown("About Us -  Displays team infor, including contact details.")
		st.image(About_Page, width=500)
		st.subheader("App Usage")
		st.markdown("App requires the user to input text (ideally a tweet relating to climate change), and will classify it according to whether or not they believe in climate change.")
		st.markdown("")
		st.markdown("Have a look at word clouds and other general exploratory data analysis on the 'EDA' page, and make predictions on the 'Prediction' page that can be navigated to in the sidebar.")
		st.markdown("In the 'Resources' page there is information about the data source and a brief data description.")
		st.subheader("Model Performance Evaluation")
		st.markdown("Model evaluation aims to estimate the generalization accuracy of a model on future (unseen) data. Methods for evaluating a model's performance use a test set (i.e data not seen by the model) to evaluate model performance. This evaluation shows total efficiency as scores.")
        
	# Building out the 'EDA' page   
	if selection == "EDA":
		st.title("Exploratory Data Analysis")   
		st.markdown("Analysis of the training data is an important step in understanding the data. A variety of analysis has been done on the training data. Select an option for more information.")
		# Building checkbox to give user options
		if st.checkbox('Hashtags for each sentiment'):
			st.markdown("See the most commonly used hashtags for tweets that:")            
        
        # Give user the choice of sentiment
			sentimentChoice = st.radio("Choose and option", ("don't believe in man-made climate change", "neither supports nor refutes the belief of man-made", "do believe in man-made climate change", "are news related to climate change"))
            
			if sentimentChoice == "don't believe in man-made climate change":
				st.image(anti_hashtags, caption='Most frequently used hashtags for tweets with negative sentiment.', use_column_width=True)
				st.markdown("It's very interesting to see that the most used hashtag here is #MAGA, which stands for 'Make America Great Again'. This was the campaign slogan used in American politics that was popularized by Donald Trump in his successful 2016 presidential campaign. This is not surprising since #trump is also frequently used for this sentiment. This supports the assumption that people who don't believe in climate change tend to support President Trump, which would make sense since he is openly sceptic about the matter.")                
            
			if sentimentChoice == "neither supports nor refutes the belief of man-made":            
				st.image(neutral_hashtags, caption='Most frequently used hashtags for tweets with a neutral sentiment.', use_column_width=True)            
				st.markdown("It's interesting to see that #trump and #beforetheflood are two of the most frequently used hashtags. However, #trump is more commonly used than #beforetheflood. #trump is prevelant in the negative sentiment and #beforetheflood is prevelant in the positive sentiment. This might allude to people leaning towards the sceptical side of the matter, just like President Trump.")
            
			if sentimentChoice == "do believe in man-made climate change":            
				st.image(pro_hashtags, caption='Most frequently used hashtags for tweets with positive sentiment.', use_column_width=True)
				st.markdown("The second most used hashtag is #beforetheflood, which refers to the documentary where actor Leonardo DiCaprio meets with scientists, activists and world leaders to discuss the dangers of climate change and possible solutions. This documentary was well received by the publc, with the majority of its audience most likely being people who believe in climate change. This assumption is supported by the fact that this hashtag was frequently used with this sentiment.")                
				st.markdown("It is interesting to notice that the positive sentiment has #cop22 and #parisagreement. This might indicate that those who believe in climate change are interested and invested in how climate change is addressed politically and internationally.") 
				st.markdown("#cop22 refers to the international political response to climate change began at the Rio Earth Summit in 1992, where the ‘Rio Convention’ included the adoption of the UN Framework on Climate Change (UNFCCC). This convention set out a framework for action aimed avoiding 'dangerous anthropogenic interference with the climate system'. The UNFCCC entered into force in 1994 and now has a near-universal membership of 196 countries. The main objective of the annual Conference of Parties (COP) is to review the Convention’s implementation.")
				st.markdown("#parisagreement refers to an agreement within the United Nations Framework Convention on Climate Change, dealing with greenhouse-gas-emissions mitigation, adaptation, and finance, signed in 2016.")
            
			if sentimentChoice == "are news related to climate change":              
				st.image(news_hashtags, caption='Most frequently used hashtags for tweets with a factual/news sentiment.', use_column_width=True)
				st.markdown("The fact that there are tweets that give the latest news or research related to climate change, shows that the public is aware of climate change and that there is scientific evidence that climate change is real and happening.")    
            
		if st.checkbox('Sentiment count analysis'):
			st.markdown("The four sentiment categories are numerically represented in the raw data.")
			st.markdown("-1: **Negative** (anti climate change)")
			st.markdown("0: **Neutral**")
			st.markdown("1: **Positive** (pro climate change)")
			st.markdown("2: **News** related")            
			st.markdown("The sentiment count for the training data is shown below.")
			st.image(unbalancedData, caption='The training data is not evenly balanced.', use_column_width=True)
			st.markdown("It is clear that the positive sentiment is the majority with over 50% if the training data fallin gin this category. Unbalanced data can lead to the model being overtrained or undertrained, giving incorrect predictions.")
			st.markdown("A total of 1965 tweets are duplicated, which is over 10% of the given data. The majority of the tweets belong to the positive sentiment category, with the most popular tweet being retweeted 306 times.")            
		if st.checkbox('Word count analysis'):
			st.markdown("Creating a word cloud to visualizate tweet keywords and text data,to highlight popular or trending terms based on frequency of use and prominence. The larger the word in the visual the more common the word is on tweet messages.")        
			st.image(wordcloud, caption='Wordclouds from the training data.', use_column_width=True)
			st.markdown("The following common words have been exluded from the wordcloud to get a better understanding of each sentiment: climate, change, RT, global, warming.")
			st.markdown("Valuable information can be gathered from these wordlouds as the words clearly display the sentiment. As an example, the negative sentiment contains words like fake, hoax, and scam.")            

	# Building out the prediction page
	if selection == "Making Predictions":
		st.title("Making prediction with machine learning models")
		st.image(classification, width=400)
		st.markdown("A machine learning model is used to classify tweets about climate change according to four categories. The categories are described as:")
		st.markdown("**Negative**: This text does not believe in man-made climate change")
		st.markdown("**Neutral**: This text neither supports nor refutes the belief of man-made climate change")
		st.markdown("**Positive**: This text supports the belief of man-made climate change")
		st.markdown("**News**: This text links to factual news about climate change")        
		st.info("Enter some text on climate change below, then choose the desired model to classify the text.")
		# Creating a text box for user input
		tweet_text = st.text_area("What's your opinion on climate change?",'')

        # cleaning text
		tweet_text = clean_tweets(tweet_text)
		tweet_text = cleaning(tweet_text)
		tweet_text = [tweet_text]
        
        # Give user the choice of more than one model
		modelChoice = st.radio("Choose a model", ("Linear SVC", "Logistic", "Ridge Classifier"))         

		if modelChoice == 'Linear SVC':
			# Loading .pkl file with the model of choice + make predictions
			predictor = joblib.load(open(os.path.join("resources/Final Models/LinearSVC.pkl"),"rb"))
			prediction = predictor.predict(tweet_text)
            
			# When model has successfully run, will print prediction
			statement(prediction)
			st.markdown("This model had a f1-accuracy score of 79% and an execution time of 1.11 seconds.")
            
		if modelChoice == 'Logistic':
			# Loading .pkl file with the model of choice + make predictions
			predictor = joblib.load(open(os.path.join("resources/Final Models/LogisticRegression.pkl"),"rb"))
			prediction = predictor.predict(tweet_text)
            
			# When model has successfully run, will print prediction
			statement(prediction)            
			st.markdown("This model had a f1-accuracy score of 77% and an execution time of 8.82 seconds.")        

		if modelChoice == 'Ridge Classifier':            
			# Loading .pkl file with the model of choice + make predictions
			predictor = joblib.load(open(os.path.join("resources/Final Models/RidgeClassifier.pkl"),"rb"))
			prediction = predictor.predict(tweet_text)
            
			# When model has successfully run, will print prediction
			statement(prediction)            
			st.markdown("This model had a f1-accuracy score of 78% and an execution time of 1.24 seconds.")

	# Building out the "Resources" page
	if selection == "Resources":
		st.title("Resources")
		st.subheader("Raw data used for training the model")
		st.markdown("Here you will find the raw twitter data that was used to train the model.")
		st.image(twitter, caption='The raw data was obtained from Twitter', width=200)
		st.markdown("The four categories are numerically represented in the raw data.")
		st.markdown("-1: Negative (anti climate change)")
		st.markdown("0: Neutral")
		st.markdown("1: Positive (pro climate change)")
		st.markdown("2: News related")        
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page            
            
	# Building the "About us" age            
	if selection == "About us":
		st.image(banner, width=700)
		st.markdown("We are six students at EXPLORE Data Science Academy, nice to meet you!")
		st.info("Anna Modjadji")
		st.image(Anna, width=250)
		st.markdown("Book worm and music enthusiast with tiny sparks of mischief")
		st.info("Buhle Ntushelo")
		st.image(Buhle, width=250)
		st.markdown("'An optimistic diplomat, lover of nature and serenity")
		st.info("Hester Stofberg")
		st.image(Hester, width=250) 
		st.markdown("Aspiring paint-by-numbers painter, plant enthusiast, and lover of all things feta.")        
		st.info("Maddy Muir")
		st.image(Maddy, width=250) 
		st.markdown("Mad about all things that are related to Maths")       
		st.info("Olwethu Mkhuhlane")
		st.image(Olwethu, width=250)
		st.markdown("Mindful individual who believes Peace is grown with a little bit more love")
		st.info("Tony Masombuka")
		st.image(Tony, width=250)
		st.markdown("An explorer of mother nature who believes in the preservation of fauna and flora")
		st.markdown("     ")
		st.image(contact, width=700)
		st.markdown("     ")
		st.info("Anna Modjadji")
		st.markdown("Github: https://github.com/AnnaM-Explore ")
		st.markdown("LinkedIn: https://www.linkedin.com/in/anna-modjadji-30b410136")
		st.info("Buhle Ntushelo")
		st.markdown("Github: https://github.com/bntushelo ")
		st.info("Hester Stofberg")
		st.markdown("Github: https://github.com/hesterstofberg ")
		st.markdown("LinkedIn: https://www.linkedin.com/in/hesterstofberg/ ")
		st.info("Maddy Muir")
		st.markdown("Github: https://github.com/Maddy-Muir ")
		st.markdown("LinkedIn: https://www.linkedin.com/in/maddy-muir-41504743/ ")
		st.info("Olwethu Mkhuhlane")
		st.markdown("Github: https://github.com/OlwethuMkhuhlane")
		st.markdown("LinkedIn: https://www.linkedin.com/mwlite/in/olwethu-mkhuhlane-9a1388113 ")
		st.info("Tony Masombuka")
		st.markdown("Github: https://github.com/TonyMasombuka ")
		st.markdown("LinkedIn: https://www.linkedin.com/in/daniel-masombuka-28547b106/ ")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
