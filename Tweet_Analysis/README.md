# Tweet Feature Engineering for Regression Analysis

This project implements feature engineering for Twitter data to predict tweet engagement metrics (retweet count). The implementation extracts text-based features, user-based features, and temporal features from tweet data.

## Files

- `tweet_feature_engineering.py`: Main class that implements feature engineering
- `test_tweet_feature_engineering.py`: Unit tests for the feature engineering class
- `extract_tweet_features.py`: Script to extract features from tweets
- `requirements.txt`: Dependencies required for the project

## Setup

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure you have the tweet data file located at `data/tweets_#superbowl_120000.pkl` or update the path in the scripts.

## Feature Engineering

The code extracts the following feature sets:

### Text-Based Features
- Tweet length (character count, word count)
- Sentiment analysis (positive, negative, neutral, compound scores)
- Tweet complexity (readability score)
- Content richness (hashtags, mentions, URLs)
- Special characters (exclamation marks, question marks)
- Media presence

### User-Based Features
- Followers count
- Author verification status
- Retweet information
- User description
- User statistics (followers, friends, statuses, etc.)
- Followers to friends ratio

### Temporal Features
- Time components (hour, day of week, month, year)
- Cyclical encoding of time features
- Weekend flag
- Time of day categories (morning, afternoon, evening, night)

## Usage

To extract features from the SuperBowl tweets dataset:

```bash
python extract_tweet_features.py --input data/tweets_#superbowl_120000.pkl --output data/superbowl_features.csv
```

Optional arguments:
- `--sample`: Number of tweets to sample (default: 0, which means all tweets)

To run tests:

```bash
python -m unittest test_tweet_feature_engineering.py
```

## Sample Workflow

The feature engineering process follows these steps:

1. Load tweet data from the pickle file
2. Extract text-based features
3. Extract user-based features
4. Extract temporal features
5. Combine all features into a single dataset
6. Save the features to a CSV file

The combined features can then be used for regression analysis to predict the number of retweets a tweet will receive.

## Future Work

- Add more advanced NLP features (topic modeling, entity recognition)
- Incorporate network features (interactions between users)
- Implement feature selection to identify the most predictive features
- Create a pipeline for model training and evaluation

## Requirements

- Python 3.7 or higher
- See `requirements.txt` for package dependencies 