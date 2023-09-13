import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import json

class Product:
    def __init__(self, name, description, category):
        self.name = name
        self.description = description
        self.category = category

class EcoProductRecommendation:
    def __init__(self):
        self.products = []
        self.recommendations = {}

    def scrape_data(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        product_names = soup.find_all('h2', class_='product-name')
        product_descs = soup.find_all('div', class_='product-desc')

        for name, desc in zip(product_names, product_descs):
            product = Product(name.text.strip(), desc.text.strip(), "Unknown")
            self.products.append(product)

    def load_dataset(self, file_path):
        self.products = pd.read_csv(file_path)
        
    def preprocess_data(self):
        texts = [product.description for product in self.products]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)

        y = [1 if 'eco-friendly' in product.description else 0 for product in self.products]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report

    def analyze_sentiment(self, reviews):
        sid = SentimentIntensityAnalyzer()
        sentiments = []
        for review in reviews:
            sentiment = sid.polarity_scores(review)
            sentiment['review'] = review
            sentiments.append(sentiment)
        return sentiments

    def get_reviews(self, product):
        reviews = []
        response = requests.get(product.url)
        soup = BeautifulSoup(response.content, 'html.parser')
        review_elements = soup.find_all('div', class_='review')
        for element in review_elements:
            review = element.text.strip()
            reviews.append(review)
        return reviews

    def recommend_products(self, user_preferences):
        self.recommendations = {}
        for category, features in user_preferences.items():
            filtered_products = [
                product for product in self.products if product.category == category]
            for feature in features:
                filtered_products = [
                    product for product in filtered_products if feature in product.description]

            recommended_products = []
            for product in filtered_products:
                reviews = self.get_reviews(product)
                sentiments = self.analyze_sentiment(reviews)
                average_sentiment = sum(sentiment['compound'] for sentiment in sentiments) / len(sentiments)
                if average_sentiment > 0.5:
                    recommended_products.append(product)

            self.recommendations[category] = recommended_products

    def visualize_carbon_savings(self):
        savings = {
            'Product A': 100,
            'Product B': 200,
            'Product C': 150,
        }
        products = list(savings.keys())
        carbon_savings = list(savings.values())
        plt.figure(figsize=(8, 6))
        sns.barplot(x=products, y=carbon_savings)
        plt.title('Carbon Savings of Eco-friendly Products')
        plt.xlabel('Product')
        plt.ylabel('Carbon Savings (tons)')
        plt.xticks(rotation=45)
        plt.show()

    def run(self, url, file_path):
        self.scrape_data(url)
        X_train, X_test, y_train, y_test = self.preprocess_data()
        model = self.train_model(X_train, y_train)
        evaluation_report = self.evaluate_model(model, X_test, y_test)
        user_preferences = {
            'Electronics': ['Sustainable materials', 'Energy efficiency'],
            'Fashion': ['Organic materials', 'Zero waste']
        }
        self.recommend_products(user_preferences)
        self.visualize_carbon_savings()
        with open('recommendations.json', 'w') as f:
            json.dump(self.recommendations, f)

        print("Program execution completed.")

eco_rec = EcoProductRecommendation()
eco_rec.run('https://example.com/products', 'dataset.csv')