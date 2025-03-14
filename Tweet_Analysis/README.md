# Twitter Engagement Regression Analysis

This project implements feature engineering and predictive modeling for Twitter data to predict tweet engagement metrics (retweet count). The implementation extracts text-based features, user-based features, and temporal features from tweet data, and applies various regression models to predict engagement.

## Files

- `tweet_feature_engineering.py`: Main class that implements feature engineering
- `test_tweet_feature_engineering.py`: Unit tests for the feature engineering class
- `extract_tweet_features.py`: Script to extract features from tweets
- `regression_model_comparison.py`: Script to train and evaluate regression models
- `tweet_regression_analysis_report.md`: Detailed analysis of modeling results
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

## Modeling Approach

We implemented and compared multiple regression and classification models to predict tweet engagement:

### Regression Models
1. **Linear Regression (without regularization)**: Baseline model to establish fundamental relationships
2. **Lasso Regression (L1 regularization)**: To perform feature selection and prevent overfitting
3. **Ridge Regression (L2 regularization)**: To handle multicollinearity in features
4. **XGBoost**: Gradient boosting model to capture non-linear relationships
5. **Neural Network (3-layer)**: Deep learning approach for complex pattern recognition

### Classification Model
- **Logistic Regression**: Binary classification model to predict whether a tweet will be retweeted at all

### Model Evaluation Metrics
For regression models:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

For classification model:
- Accuracy
- Precision/Recall
- F1 Score

## Modeling Results

### Regression Performance

| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| Neural Network (3 layers) | 1.6956 | 0.0043 | 0.2101 |
| XGBoost | 1.6972 | 0.0024 | 0.1353 |
| Lasso Regression (α=0.001) | 1.6993 | -0.0001 | 0.1513 |
| Linear/Ridge Regression | 1.6995 | -0.0004 | ~0.154 |

### Classification Performance
Logistic Regression achieved 96.72% accuracy for predicting whether a tweet would be retweeted, but with very low recall for the positive class.

### Key Findings
1. All regression models achieved R² scores close to zero, indicating that exact retweet count prediction is extremely challenging
2. Neural Network and XGBoost slightly outperformed linear models
3. The most important features across models were:
   - User influence metrics (follower counts)
   - Content characteristics (tweet length, sentiment)
   - Temporal features (hour of day, day of week)
4. The extreme class imbalance (only 3.35% of tweets receive any retweets) poses significant modeling challenges

## Usage

To extract features from the SuperBowl tweets dataset:

```bash
python extract_tweet_features.py --input data/tweets_#superbowl_120000.pkl --output data/superbowl_features.csv
```

Optional arguments:
- `--sample`: Number of tweets to sample (default: 0, which means all tweets)

To run the regression model comparison:

```bash
python regression_model_comparison.py
```

To run tests:

```bash
python -m unittest test_tweet_feature_engineering.py
```

## Sample Workflow

1. **Feature Engineering**:
   - Load tweet data from the pickle file
   - Extract text-based, user-based, and temporal features
   - Combine all features into a single dataset
   - Save the features to a CSV file

2. **Model Training & Evaluation**:
   - Load the features dataset
   - Preprocess and split into training/testing sets
   - Train multiple regression models
   - Evaluate and compare model performance
   - Generate visualizations of results

## Challenges & Future Work

- **Address Class Imbalance**: Use techniques like oversampling, SMOTE, or weighted loss functions
- **Advanced Feature Engineering**: Add topic modeling, word embeddings, or network-based features
- **Model Enhancement**: Try ensemble methods, more sophisticated neural network architectures, or two-stage prediction approaches
- **Alternative Formulations**: Consider zero-inflated models or ordinal regression with retweet count buckets
- **Incorporate External Context**: Add features related to trending topics and events occurring during the tweet's lifetime

## Requirements

- Python 3.7 or higher
- See `requirements.txt` for package dependencies 