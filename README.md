# Predicting the total game score of AFL matches - Proof of Concept

This script aims to predict the total game score of AFL matches using publicly available data.

The purpose of the project is to quickly understand the predictability of AFL match results using a number of likely influences, while also practicing the implementation of OOO principles. This would determine if sourcing additional data to enhance the predictions would be worthwhile.

The model is built using Scikit-Learn's Random Forest Classifier, employing a Grid Search to tune the hyperparameters.

Data sourced from https://www.kaggle.com/datasets/stoney71/aflstats.

A note on the model results:
1. The model's accuracy when predicting which bucket the total game score would fall into, was around 34%. However the predict_proba method was used instead to firstly calculate the likelihood of the score falling into each bucket, then using a cumulative percentage rule to determine an appropriate over/under score to improve the output. The accuracy of the latter approach was around 86%.
2. Here are the bins used to create each bucket. [0, 120, 150, 180, 210, 240, 300]
3. With the greatly improved model accuracy, the model often makes conservative predictions of little value. The opportunity therefore lies within predictions that more closely reflect total score line betting odds.
4. One of the features, rainfall in mm is obviously not known until after a game is played. The workaround was to create a rainfall category that could reasonably be replicated using rainfall forecasts on the day.
