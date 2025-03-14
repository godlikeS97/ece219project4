import pandas as pd
import numpy as np
import re
import pickle
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import textstat
from sklearn.feature_extraction.text import CountVectorizer
import json
from dateutil import parser


class TweetFeatureEngineering:
    """
    A class to load tweet data and extract various features for regression analysis.
    """
    
    def __init__(self, pickle_file_path):
        """
        Initialize the TweetFeatureEngineering object.
        
        Args:
            pickle_file_path (str): Path to the pickle file containing tweet data.
        """
        self.pickle_file_path = pickle_file_path
        self.tweets_df = None
        self.features_df = None
        
        # Download required NLTK packages if not already downloaded
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        try:
            nltk.data.find('punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Fix for punkt_tab error in newer NLTK versions
        try:
            nltk.data.find('tokenizers/punkt_tab/english/')
        except LookupError:
            nltk.download('punkt')
            # If punkt_tab is not available, we'll use a workaround for word_tokenize
        
        # Initialize sentiment analyzer
        self.sid = SentimentIntensityAnalyzer()
    
    def load_data(self):
        """
        Load tweet data from pickle file.
        
        Returns:
            pandas.DataFrame: DataFrame containing tweet data.
        """
        print(f"Loading data from {self.pickle_file_path}...")
        self.tweets_df = pd.read_pickle(self.pickle_file_path)
        print(f"Loaded {len(self.tweets_df)} tweets.")
        return self.tweets_df
    
    def extract_text_features(self):
        """
        Extract text-based features from tweets.
        
        Returns:
            pandas.DataFrame: DataFrame with text-based features.
        """
        if self.tweets_df is None:
            self.load_data()
        
        print("Extracting text-based features...")
        
        # Create a DataFrame to store text features
        text_features = pd.DataFrame(index=self.tweets_df.index)
        
        # Get tweet text
        tweets = self.tweets_df['tweet_text'].fillna('')
        
        # 1. Tweet length features
        text_features['tweet_char_count'] = tweets.str.len()
        
        # Use a safe word tokenize function that doesn't rely on punkt_tab
        def safe_tokenize(text):
            if not isinstance(text, str) or not text.strip():
                return 0
            # Simple tokenization for robustness
            return len(text.split())
        
        text_features['tweet_word_count'] = tweets.apply(safe_tokenize)
        
        # 2. Sentiment analysis
        text_features['sentiment_compound'] = tweets.apply(lambda x: self.sid.polarity_scores(x)['compound'])
        text_features['sentiment_positive'] = tweets.apply(lambda x: self.sid.polarity_scores(x)['pos'])
        text_features['sentiment_negative'] = tweets.apply(lambda x: self.sid.polarity_scores(x)['neg'])
        text_features['sentiment_neutral'] = tweets.apply(lambda x: self.sid.polarity_scores(x)['neu'])
        
        # 3. Tweet complexity
        text_features['readability_score'] = tweets.apply(lambda x: textstat.flesch_reading_ease(x) if len(x) > 0 else 0)
        
        # 4. Content richness
        text_features['hashtag_count'] = tweets.apply(lambda x: x.count('#') if isinstance(x, str) else 0)
        text_features['mention_count'] = tweets.apply(lambda x: x.count('@') if isinstance(x, str) else 0)
        
        # Use a simpler approach for URL count to avoid regex issues
        text_features['url_count'] = tweets.apply(lambda x: x.count('http') if isinstance(x, str) else 0)
        
        # 5. Special characters and features
        text_features['exclamation_count'] = tweets.apply(lambda x: x.count('!') if isinstance(x, str) else 0)
        text_features['question_count'] = tweets.apply(lambda x: x.count('?') if isinstance(x, str) else 0)
        text_features['capital_ratio'] = tweets.apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if isinstance(x, str) and len(x) > 0 else 0)
        
        # 6. Has media flag from tweet_entities
        def has_media(entities):
            if isinstance(entities, str):
                try:
                    entities_dict = json.loads(entities.replace("'", '"'))
                    return 1 if 'media' in entities_dict and len(entities_dict['media']) > 0 else 0
                except:
                    return 0
            return 0
        
        text_features['has_media'] = self.tweets_df['tweet_entities'].apply(has_media)
        
        print(f"Extracted {len(text_features.columns)} text-based features.")
        return text_features
    
    def extract_user_features(self):
        """
        Extract user-based features from tweets.
        
        Returns:
            pandas.DataFrame: DataFrame with user-based features.
        """
        if self.tweets_df is None:
            self.load_data()
        
        print("Extracting user-based features...")
        
        # Create a DataFrame to store user features
        user_features = pd.DataFrame(index=self.tweets_df.index)
        
        # 1. Followers count
        user_features['author_followers_count'] = pd.to_numeric(self.tweets_df['author_followers'], errors='coerce').fillna(0)
        
        # 2. Author type
        user_features['is_verified'] = self.tweets_df['author_type'].apply(lambda x: 1 if x == 'verified' else 0)
        
        # 3. Original author features
        original_followers = pd.to_numeric(self.tweets_df['original_author_followers'], errors='coerce').fillna(0)
        user_features['original_author_followers_count'] = original_followers
        
        # 4. Is retweet
        user_features['is_retweet'] = self.tweets_df['original_author_name'].notna().astype(int)
        
        # 5. User description length
        user_features['author_description_length'] = self.tweets_df['author_description'].fillna('').str.len()
        
        # 6. Extract user features from tweet_user field
        def extract_user_info(user_json):
            if isinstance(user_json, str):
                try:
                    user_dict = json.loads(user_json.replace("'", '"'))
                    followers = user_dict.get('followers_count', 0)
                    friends = user_dict.get('friends_count', 0)
                    statuses = user_dict.get('statuses_count', 0)
                    listed = user_dict.get('listed_count', 0)
                    verified = 1 if user_dict.get('verified', False) else 0
                    
                    return pd.Series([
                        followers, 
                        friends, 
                        statuses, 
                        listed, 
                        verified,
                        followers / (friends + 1)  # followers to friends ratio
                    ])
                except:
                    return pd.Series([0, 0, 0, 0, 0, 0])
            return pd.Series([0, 0, 0, 0, 0, 0])
        
        user_info = self.tweets_df['tweet_user'].apply(extract_user_info)
        user_info.columns = [
            'user_followers_count', 
            'user_friends_count', 
            'user_statuses_count', 
            'user_listed_count', 
            'user_verified',
            'followers_friends_ratio'
        ]
        
        # Combine with user_features
        user_features = pd.concat([user_features, user_info], axis=1)
        
        print(f"Extracted {len(user_features.columns)} user-based features.")
        return user_features
    
    def extract_temporal_features(self):
        """
        Extract temporal features from tweets.
        
        Returns:
            pandas.DataFrame: DataFrame with temporal features.
        """
        if self.tweets_df is None:
            self.load_data()
        
        print("Extracting temporal features...")
        
        # Create a DataFrame to store temporal features
        temporal_features = pd.DataFrame(index=self.tweets_df.index)
        
        # Parse created_at timestamps
        def parse_created_at(time_str):
            try:
                return parser.parse(time_str)
            except:
                return None
        
        created_at = self.tweets_df['tweet_created_at'].apply(parse_created_at)
        
        # Extract basic time components
        temporal_features['hour_of_day'] = created_at.dt.hour
        temporal_features['day_of_week'] = created_at.dt.dayofweek
        temporal_features['month'] = created_at.dt.month
        temporal_features['year'] = created_at.dt.year
        
        # Cyclical encoding of time features
        temporal_features['hour_sin'] = np.sin(2 * np.pi * temporal_features['hour_of_day'] / 24)
        temporal_features['hour_cos'] = np.cos(2 * np.pi * temporal_features['hour_of_day'] / 24)
        temporal_features['day_sin'] = np.sin(2 * np.pi * temporal_features['day_of_week'] / 7)
        temporal_features['day_cos'] = np.cos(2 * np.pi * temporal_features['day_of_week'] / 7)
        
        # Is weekend
        temporal_features['is_weekend'] = temporal_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Time of day categories
        hour = temporal_features['hour_of_day']
        temporal_features['is_morning'] = ((hour >= 5) & (hour < 12)).astype(int)
        temporal_features['is_afternoon'] = ((hour >= 12) & (hour < 17)).astype(int)
        temporal_features['is_evening'] = ((hour >= 17) & (hour < 22)).astype(int)
        temporal_features['is_night'] = (((hour >= 22) & (hour < 24)) | ((hour >= 0) & (hour < 5))).astype(int)
        
        print(f"Extracted {len(temporal_features.columns)} temporal features.")
        return temporal_features
    
    def combine_all_features(self):
        """
        Combine all extracted features into a single DataFrame.
        
        Returns:
            pandas.DataFrame: DataFrame with all extracted features.
        """
        print("Combining all features...")
        
        # Extract all feature sets
        text_features = self.extract_text_features()
        user_features = self.extract_user_features()
        temporal_features = self.extract_temporal_features()
        
        # Combine all features
        self.features_df = pd.concat([text_features, user_features, temporal_features], axis=1)
        
        # Add the target variable
        self.features_df['retweet_count'] = pd.to_numeric(self.tweets_df['tweet_retweet_count'], errors='coerce').fillna(0)
        
        print(f"Created combined feature set with {len(self.features_df.columns)} features.")
        return self.features_df
    
    def save_features(self, output_path):
        """
        Save the extracted features to a CSV file.
        
        Args:
            output_path (str): Path to save the features CSV file.
        """
        if self.features_df is None:
            self.combine_all_features()
            
        print(f"Saving features to {output_path}...")
        self.features_df.to_csv(output_path, index=False)
        print("Features saved successfully.") 