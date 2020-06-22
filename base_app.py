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

# Vectorizer
news_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

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
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Raw data" page
	if selection == "Information":
		st.info("Data used for training the model")
		# You can read a markdown file from supporting resources folder
		st.markdown("Here you will find the raw data that was used to train the model, in order to make some predictions.")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page
            
	# Building out the 'Background' page
	if selection == "Background":
		st.info("How it works")
		st.markdown("This web app requires the user to input text (ideally a tweet relating to climate change), and will classify it according to whether or not they believe in climate change. You can have a look at word clouds and other general exloratory data analysis on the 'EDA' page, and make your predictions on the 'Prediction' page that you can navigate to in the sidebar. In the 'Information' page you will find information about the data source and a brief data description.")
        
	# Building out the 'EDA' page   
	if selection == "EDA":
		st.info("Word count analysis")
		st.markdown("below you will see the wordclouds")
		# here we can add graphs and word clouds and such        
		st.image(wordcloud, caption='Wordcloud from the training data.', use_column_width=True)
		st.info("Word frequency analysis")
		# write something here
		st.info("Word count analysis")
		# write something here
		st.image(wordcount, caption='Top 20 most frequently used words.', use_column_width=True)
		st.info("Hashtags for each sentiment")
		# write something here
		st.info("Average length of each sentiment")


	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with machine learning models")
		st.markdown("A machine learning model is used to classify tweets about climate change according to three categories.")
		st.table(pd.DataFrame({'Category': [-1, 0, 1, 2],'Description': ['Anti: this tweet does not believe in man-made climate change', 'Neutral: this tweet neither supports nor refutes the belief of man-made climate change', 'Pro: this tweet supports the belief of man-made climate change', 'News: this tweet links to factual news about climate change']}))
		# Creating a text box for user input
		tweet_text = st.text_area("What's your opinion on climate change?",'')
		# add function to clean text

		if st.button("Analyse my opinion"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.
			st.success("Your opinion has been categorized by the model as: {}".format(prediction))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
