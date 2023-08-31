import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# Download the VADER lexicon for sentiment analysis (only needs to be done once)
nltk.download('vader_lexicon')

# Load dataset
data = pd.read_csv('C:/Users/HP/OneDrive/Desktop/SentimentAnalysisProject/data/sentiment_data.csv')

# Initialize the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Preprocess the text data
data['Text'] = data['Text'].apply(lambda text: text.strip())  # Remove leading/trailing spaces

# Analyze sentiment using VADER for all rows
data['Sentiment_Scores'] = data['Text'].apply(lambda text: sia.polarity_scores(text))

# Classify sentiment as positive, negative, or neutral based on compund score
data['Predicted_Sentiment'] = data['Sentiment_Scores'].apply(lambda scores: "Positive" if scores['compound'] > 0.1 else ("Negative" if scores['compound'] < -0.1 else "Neutral"))

# Print the text and predicted sentiment for each row
print(data[['Text', 'Predicted_Sentiment']])

# Create a bar chart to visualize the distribution of predicted sentiments
sentiment_counts = data['Predicted_Sentiment'].value_counts()
sentiment_counts.plot(kind='bar')
plt.title("Distribution of Predicted Sentiments")
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
