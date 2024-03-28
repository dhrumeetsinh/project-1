from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
import requests
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class TwitterSentimentApp(App):
    def build(self):
        self.query = ''
        self.sia = SentimentIntensityAnalyzer()
        
        layout = BoxLayout(orientation='vertical')
        
        query_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=50)
        self.query_input = TextInput(text='#Oscar2024')
        query_layout.add_widget(Label(text="Enter Query: "))
        query_layout.add_widget(self.query_input)
        layout.add_widget(query_layout)

        button = Button(text="Get Tweets and Analyze Sentiment")
        button.bind(on_press=self.on_button_press)
        layout.add_widget(button)
        
        self.output_label = Label(text="Waiting for tweets...")
        layout.add_widget(self.output_label)

        return layout

    def on_button_press(self, instance):
        self.query = self.query_input.text
        self.get_tweets_and_analyze_sentiment()

    def classify_sentiment(self, score):
        if score >= 0.05:
            return 'positive'
        elif score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def get_sentiment(self, text):
        return self.sia.polarity_scores(text)['compound']

    def process_tweets(self, df):
        df['sentiment_score'] = df['Text'].apply(self.get_sentiment)
        df['sentiment'] = df['sentiment_score'].apply(self.classify_sentiment)
        return df

    def get_tweets_and_analyze_sentiment(self):
        url = "https://twitter241.p.rapidapi.com/search"
        querystring = {"type": "Latest", "count": "1000", "query": self.query}
        headers = {
            "X-RapidAPI-Key": "c80958fbe9msh337718e000fa802p140f81jsn5f34a969d758",
            "X-RapidAPI-Host": "twitter241.p.rapidapi.com"
        }

        response = requests.get(url, headers=headers, params=querystring)

        if response.status_code == 200:
            data = response.json()
            tweets = []

            for tweet in data.get('result', {}).get('timeline', {}).get('instructions', []):
                for entry in tweet.get('entries', []):
                    tweet_content = entry.get('content', {}).get('itemContent', {}).get('__typename', '')

                    if tweet_content == 'TimelineTweet':
                        tweet_data = entry.get('content', {}).get('itemContent', {}).get('tweet_results', {}).get('result', {})
                        user_data = tweet_data.get('core', {}).get('user_results', {}).get('result', {})

                        tweet_text = tweet_data.get('legacy', {}).get('full_text', '')
                        user_name = user_data.get('legacy', {}).get('name', '')
                        tweet_date = tweet_data.get('legacy', {}).get('created_at', '')

                        tweets.append({
                            'User': user_name,
                            'Date': tweet_date,
                            'Text': tweet_text
                        })

            df = pd.DataFrame(tweets)
            df = self.process_tweets(df)

            all_text = ' '.join(df['Text'])
            self.generate_wordcloud(all_text)
            self.plot_histogram(df)

        else:
            self.output_label.text = f"Error: {response.status_code} - {response.text}"

    def generate_wordcloud(self, text):
        wordcloud = WordCloud(width=800, height=400, random_state=21, max_font_size=110).generate(text)
        
        plt.figure(figsize=(10, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.show()

    def plot_histogram(self, df):
        plt.figure(figsize=(8, 6))
        df['sentiment'].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

if __name__ == '__main__':
    TwitterSentimentApp().run()
