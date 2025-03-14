import os
import sys
import unittest
import pandas as pd
import numpy as np
from tweet_feature_engineering import TweetFeatureEngineering

class TestTweetFeatureEngineering(unittest.TestCase):
    """
    Test cases for the TweetFeatureEngineering class.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment once before all tests.
        """
        # Define the path to the pickle file
        cls.pickle_file_path = os.path.join('data', 'tweets_#superbowl_120000.pkl')
        
        # Check if the pickle file exists
        if not os.path.exists(cls.pickle_file_path):
            # For testing with a smaller dataset, create a sample DataFrame
            cls.use_sample_data = True
            print(f"Warning: {cls.pickle_file_path} not found. Using sample data for testing.")
            
            # Create sample data
            cls.sample_tweets = pd.DataFrame({
                'tweet_text': [
                    "I love the #SuperBowl! Amazing game @NFL http://example.com",
                    "Can't wait for the #SuperBowl tonight! #GoPatriots",
                    "This game is horrible... #SuperBowl #disappointed",
                    ""
                ],
                'author_followers': ['1000', '500', '250', '100'],
                'author_type': ['verified', 'normal', 'normal', 'normal'],
                'original_author_followers': ['2000', None, None, None],
                'original_author_name': ['@someceleb', None, None, None],
                'author_description': ['Sports fan', 'Football lover', None, 'Just a regular user'],
                'tweet_entities': [
                    "{'hashtags': [{'text': 'SuperBowl'}], 'media': [{'type': 'photo'}]}",
                    "{'hashtags': [{'text': 'SuperBowl'}, {'text': 'GoPatriots'}]}",
                    "{'hashtags': [{'text': 'SuperBowl'}, {'text': 'disappointed'}]}",
                    "{}"
                ],
                'tweet_user': [
                    "{'followers_count': 1200, 'friends_count': 500, 'statuses_count': 5000, 'listed_count': 20, 'verified': True}",
                    "{'followers_count': 600, 'friends_count': 400, 'statuses_count': 2000, 'listed_count': 5, 'verified': False}",
                    "{'followers_count': 300, 'friends_count': 200, 'statuses_count': 1000, 'listed_count': 2, 'verified': False}",
                    "{'followers_count': 50, 'friends_count': 100, 'statuses_count': 500, 'listed_count': 0, 'verified': False}"
                ],
                'tweet_created_at': [
                    'Sun Feb 01 18:30:00 +0000 2015',
                    'Sun Feb 01 20:15:00 +0000 2015',
                    'Sun Feb 01 22:45:00 +0000 2015',
                    'Mon Feb 02 01:30:00 +0000 2015'
                ],
                'tweet_retweet_count': ['100', '50', '20', '0']
            })
            
            # Create a temporary pickle file for testing
            cls.temp_pickle_path = 'data/temp_test_tweets.pkl'
            os.makedirs(os.path.dirname(cls.temp_pickle_path), exist_ok=True)
            cls.sample_tweets.to_pickle(cls.temp_pickle_path)
            cls.feature_eng = TweetFeatureEngineering(cls.temp_pickle_path)
        else:
            # Use actual data
            cls.use_sample_data = False
            cls.feature_eng = TweetFeatureEngineering(cls.pickle_file_path)
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up after all tests.
        """
        # Remove temporary pickle file if created
        if hasattr(cls, 'temp_pickle_path') and os.path.exists(cls.temp_pickle_path):
            os.remove(cls.temp_pickle_path)
    
    def test_load_data(self):
        """
        Test if data is loaded correctly.
        """
        df = self.feature_eng.load_data()
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)
        
        # Check if the DataFrame has the expected columns
        expected_columns = [
            'tweet_text', 'author_followers', 'author_type', 
            'original_author_followers', 'original_author_name',
            'tweet_entities', 'tweet_user', 'tweet_created_at', 'tweet_retweet_count'
        ]
        
        for col in expected_columns:
            if col not in df.columns and not self.use_sample_data:
                print(f"Warning: Expected column '{col}' not found in the actual data.")
    
    def test_extract_text_features(self):
        """
        Test text feature extraction.
        """
        text_features = self.feature_eng.extract_text_features()
        self.assertIsNotNone(text_features)
        self.assertIsInstance(text_features, pd.DataFrame)
        
        # Check if the expected text features are present
        expected_features = [
            'tweet_char_count', 'tweet_word_count',
            'sentiment_compound', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral',
            'readability_score', 'hashtag_count', 'mention_count', 'url_count',
            'exclamation_count', 'question_count', 'capital_ratio', 'has_media'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, text_features.columns,
                          f"Expected text feature '{feature}' not found")
    
    def test_extract_user_features(self):
        """
        Test user feature extraction.
        """
        user_features = self.feature_eng.extract_user_features()
        self.assertIsNotNone(user_features)
        self.assertIsInstance(user_features, pd.DataFrame)
        
        # Check if the expected user features are present
        expected_features = [
            'author_followers_count', 'is_verified', 
            'original_author_followers_count', 'is_retweet',
            'author_description_length', 'user_followers_count',
            'user_friends_count', 'user_statuses_count',
            'user_listed_count', 'user_verified', 'followers_friends_ratio'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, user_features.columns,
                          f"Expected user feature '{feature}' not found")
    
    def test_extract_temporal_features(self):
        """
        Test temporal feature extraction.
        """
        temporal_features = self.feature_eng.extract_temporal_features()
        self.assertIsNotNone(temporal_features)
        self.assertIsInstance(temporal_features, pd.DataFrame)
        
        # Check if the expected temporal features are present
        expected_features = [
            'hour_of_day', 'day_of_week', 'month', 'year',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend',
            'is_morning', 'is_afternoon', 'is_evening', 'is_night'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, temporal_features.columns,
                          f"Expected temporal feature '{feature}' not found")
    
    def test_combine_all_features(self):
        """
        Test combining all features.
        """
        combined_features = self.feature_eng.combine_all_features()
        self.assertIsNotNone(combined_features)
        self.assertIsInstance(combined_features, pd.DataFrame)
        
        # Check if the target variable is included
        self.assertIn('retweet_count', combined_features.columns,
                     "Target variable 'retweet_count' not found in combined features")
        
        # Check if the combined DataFrame has all the expected features
        if self.use_sample_data:
            expected_min_column_count = 38  # Based on our implementation
            self.assertGreaterEqual(len(combined_features.columns), expected_min_column_count,
                                   f"Expected at least {expected_min_column_count} columns, got {len(combined_features.columns)}")
    
    def test_save_features(self):
        """
        Test saving features to a file.
        """
        # Create a temporary output path
        temp_output_path = 'data/test_features_output.csv'
        
        # Save the features
        self.feature_eng.save_features(temp_output_path)
        
        # Check if the file was created
        self.assertTrue(os.path.exists(temp_output_path),
                        f"Feature output file {temp_output_path} was not created")
        
        # Verify the saved file can be read back
        saved_features = pd.read_csv(temp_output_path)
        self.assertIsNotNone(saved_features)
        self.assertIsInstance(saved_features, pd.DataFrame)
        
        # Clean up
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)


if __name__ == '__main__':
    unittest.main() 