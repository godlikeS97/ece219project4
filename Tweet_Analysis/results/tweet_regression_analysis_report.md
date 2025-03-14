# Twitter Engagement Regression Analysis Report

## Project Overview

This report summarizes the results of a regression analysis project aimed at predicting tweet engagement metrics (retweet count) from the SuperBowl tweets dataset. The analysis compares several regression models and evaluates their performance in predicting how many retweets a tweet will receive.

## Data Description

The dataset consists of 120,000 tweets related to the SuperBowl, with 38 features extracted through our feature engineering process. These features fall into three main categories:

1. **Text-Based Features**: Sentiment scores, tweet length, readability, hashtag/mention counts, etc.
2. **User-Based Features**: Follower counts, verification status, user description length, etc.
3. **Temporal Features**: Hour of day, day of week, cyclical time encoding, etc.

### Target Variable Distribution

- **Mean**: 0.07 retweets per tweet
- **Median**: 0.00 retweets per tweet (most tweets receive no retweets)
- **Max**: 309.00 retweets (most retweeted tweet)
- **Tweets with at least one retweet**: 3.35% (4,020 tweets)

This shows that retweet behavior is highly skewed, with the vast majority of tweets receiving no retweets at all. This imbalance presents a challenge for regression models.

## Model Comparison

We implemented and compared six different regression models:

1. **Linear Regression (without regularization)**
2. **Lasso Regression (L1 regularization)**
3. **Ridge Regression (L2 regularization)**
4. **Logistic Regression** (for binary classification: retweet vs. no retweet)
5. **XGBoost**
6. **Neural Network** (3-layer architecture)

### Regression Performance Metrics

| Model | MSE | RMSE | MAE | R² |
|-------|-----|------|-----|-----|
| Linear Regression | 2.8885 | 1.6995 | 0.1540 | -0.0004 |
| Lasso Regression (α=0.001) | 2.8877 | 1.6993 | 0.1513 | -0.0001 |
| Ridge Regression (α=100.0) | 2.8884 | 1.6995 | 0.1539 | -0.0003 |
| XGBoost | 2.8805 | 1.6972 | 0.1353 | 0.0024 |
| Neural Network (3 layers) | 2.8751 | 1.6956 | 0.2101 | 0.0043 |

### Binary Classification Performance (Logistic Regression)

- **Accuracy**: 96.72%
- **Precision (Class 1)**: 50.00%
- **Recall (Class 1)**: 0.00%
- **F1-Score (Class 1)**: 0.01%

## Key Findings

1. **Overall Model Performance**:
   - All regression models performed similarly with R² scores close to zero, indicating that predicting the exact number of retweets is extremely challenging
   - The Neural Network and XGBoost slightly outperformed linear models
   - The high accuracy but near-zero recall for the positive class in logistic regression shows the model is primarily predicting "no retweets" due to class imbalance

2. **Most Important Features**:

   **For Linear Models**:
   - Author follower count
   - Time-related features (day cosine, hour cosine, day of week)
   - Sentiment compound score
   - URL count

   **For XGBoost**:
   - Tweet character count
   - Sentiment compound score
   - Author description length
   - Readability score
   - Tweet word count

3. **Feature Importance Patterns**:
   - User influence metrics (followers) consistently appear as important
   - Content characteristics (length, sentiment) are strong predictors
   - Temporal features show moderate importance across models

## Challenges and Limitations

1. **Extreme Class Imbalance**: With only 3.35% of tweets receiving any retweets, models struggle to identify patterns for successful tweets

2. **Non-linear Relationships**: Standard linear models fail to capture complex non-linear relationships between features and retweet counts

3. **Limited Predictive Power**: Even the best models achieved R² scores close to zero, suggesting either:
   - Missing important predictive features
   - Random/viral nature of retweet behavior that's inherently unpredictable
   - Need for more sophisticated modeling approaches

4. **Binary Classification Approach**: The logistic regression model achieved high accuracy but failed to identify any positive cases, making it ineffective despite its seemingly good performance

## Conclusions

1. Predicting the exact number of retweets for tweets is extremely challenging, as evidenced by the low R² scores across all models.

2. The Neural Network and XGBoost models marginally outperformed linear models, suggesting there are non-linear relationships in the data that more complex models can partially capture.

3. The most important predictive features relate to:
   - User influence (follower counts)
   - Tweet content (length, sentiment, readability)
   - Timing (hour of day, day of week)

4. The extreme class imbalance makes it difficult to predict which tweets will be retweeted at all, with most models defaulting to predicting zero retweets.

## Future Work

1. **Address Class Imbalance**: Use techniques like oversampling, SMOTE, or weighted loss functions to better handle the imbalanced dataset

2. **Feature Engineering**:
   - Extract more advanced text features using topic modeling or embeddings
   - Incorporate network effects and interaction history
   - Add external context features (trending topics, related events)

3. **Model Improvements**:
   - Try ensemble methods combining multiple model types
   - Implement more sophisticated neural network architectures
   - Consider two-stage models: first classify if a tweet will be retweeted, then predict how many retweets

4. **Alternative Approaches**:
   - Frame the problem as an ordinal classification task with retweet count buckets
   - Use zero-inflated models specifically designed for data with excessive zeros
   - Consider time-series approaches to model viral spread patterns 