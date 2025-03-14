#!/usr/bin/env python3
"""
Script to extract features from tweets using the TweetFeatureEngineering class.
"""

import os
import sys
import argparse
from tweet_feature_engineering import TweetFeatureEngineering

def main():
    """
    Main function to extract features from tweets.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract features from tweets')
    parser.add_argument('--input', type=str, default='data/tweets_#superbowl_120000.pkl',
                        help='Path to the input pickle file')
    parser.add_argument('--output', type=str, default='data/superbowl_features.csv',
                        help='Path to the output CSV file')
    parser.add_argument('--sample', type=int, default=0,
                        help='Number of tweets to sample (0 for all)')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        return 1
    
    # Initialize feature engineering
    print(f"Initializing feature engineering for {args.input}...")
    feature_eng = TweetFeatureEngineering(args.input)
    
    # Load data
    tweets_df = feature_eng.load_data()
    
    # Sample data if requested
    if args.sample > 0 and len(tweets_df) > args.sample:
        print(f"Sampling {args.sample} tweets from {len(tweets_df)} total tweets...")
        tweets_df = tweets_df.sample(n=args.sample, random_state=42)
        feature_eng.tweets_df = tweets_df
    
    # Extract features
    print("\n=== Feature Extraction Summary ===")
    
    print("\n1. Extracting text-based features...")
    text_features = feature_eng.extract_text_features()
    print(f"   - Generated {len(text_features.columns)} text features")
    print(f"   - Sample features: {', '.join(list(text_features.columns)[:5])}")
    
    print("\n2. Extracting user-based features...")
    user_features = feature_eng.extract_user_features()
    print(f"   - Generated {len(user_features.columns)} user features")
    print(f"   - Sample features: {', '.join(list(user_features.columns)[:5])}")
    
    print("\n3. Extracting temporal features...")
    temporal_features = feature_eng.extract_temporal_features()
    print(f"   - Generated {len(temporal_features.columns)} temporal features")
    print(f"   - Sample features: {', '.join(list(temporal_features.columns)[:5])}")
    
    print("\n4. Combining all features...")
    combined_features = feature_eng.combine_all_features()
    print(f"   - Total features: {len(combined_features.columns)}")
    
    # Save features
    print(f"\nSaving all features to {args.output}...")
    feature_eng.save_features(args.output)
    print("Feature extraction complete!")
    
    # Print some basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Total tweets: {len(tweets_df)}")
    print(f"Total features: {len(combined_features.columns)}")
    
    retweet_count = combined_features['retweet_count']
    print(f"Retweet count statistics:")
    print(f"   - Mean: {retweet_count.mean():.2f}")
    print(f"   - Median: {retweet_count.median():.2f}")
    print(f"   - Max: {retweet_count.max():.2f}")
    print(f"   - Min: {retweet_count.min():.2f}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 