import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

class Model:

    def __init__(self, model_df, target_var: str):
        self.model_df = model_df
        self.target_var = target_var

    def data_prep(self):
        X = self.model_df.drop(self.target_var, axis=1)
        y = self.model_df[self.target_var]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test


class TotalScoreRF(Model):
    max_features_range = np.arange(1, 6, 1)
    n_estimators_range = np.arange(20, 420, 20)
    param_grid = {
        'bootstrap': [True],
        'max_depth': [80, 90, 100, 110],
        'max_features': max_features_range,
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': n_estimators_range
    }

    def __init__(self, model_df, target_var: str, bins: list, save_path: str, file_name: str):
        super().__init__(
            model_df, target_var
        )

        self.bins = bins
        
        self.save_path = save_path

        self.file_name = file_name

        self.file_path = f'{self.save_path}\\{self.file_name}'

        self.X_train, self.X_test, self.y_train, self.y_test = self.data_prep()

    def grid_search_train(self):
        # Create a random forest model
        rf = RandomForestClassifier(random_state=42)

        # Tune hyperparameters using GridSearch
        grid = GridSearchCV(estimator=rf, param_grid=self.param_grid, cv=5, n_jobs=-1, verbose=2)

        grid.fit(self.X_train, self.y_train)

        print(f'The best params are {grid.best_params_} with a score of {grid.best_score_}')

        rf_model = grid.best_estimator_

        pickle.dump(rf_model, open(self.file_path, "wb"))

    def make_base_predictions(self):
        with open(self.file_path, 'rb') as file:
            rf_model = pickle.load(file)
        
        # Make predictions on the testing data
        y_pred = rf_model.predict(self.X_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(self.y_test, y_pred)

        print('Raw bin prediction accuracy:', accuracy)

    def make_adj_predictions(self):
        with open(self.file_path, 'rb') as file:
            rf_model = pickle.load(file)
        
        y_pred_prob = rf_model.predict_proba(self.X_test)

        y_pred_prob_df = pd.DataFrame(y_pred_prob, columns=self.bins[1:])

        y_pred_prob_df['label_code'] = self.y_test.tolist()

        # Produce adjusted predictions
        bins_adj = self.bins[1:]

        labels = np.arange(1, len(self.bins))

        score_labels = dict(zip(labels, bins_adj))

        y_pred_prob_df['label'] = y_pred_prob_df['label_code'].map(score_labels)

        @staticmethod
        def get_predicted_columns(row):
            threshold = 0.65
            values = row.values[:-2]

            if values[0] > threshold:
                params = [0, 'Under']
            elif np.sum(values[:2]) > threshold:
                params = [1, 'Under']
            elif np.sum(values[:3]) > threshold:
                params = [2, 'Under']
            elif np.sum(values[:4]) > np.sum(values[-4:]):
                params = [3, 'Under']
            else:
                params = [2, 'Over']
            
            new_series = pd.Series([row.index[params[0]], params[1]], index=['predicted_limit', 'predicted_direction'])
            return new_series
        
        new_columns = y_pred_prob_df.apply(get_predicted_columns, axis=1)

        y_pred_prob_df[['predicted_limit', 'predicted_direction']] = new_columns

        @staticmethod
        def run_prediction_test(row):
            if row['predicted_direction'] == 'Under':
                return row['predicted_limit'] >= row['label']
            else:
                return row['predicted_limit'] < row['label']

        y_pred_prob_df['correct'] = y_pred_prob_df.apply(run_prediction_test, axis=1)

        print('Proportion of correct predictions:')

        print(y_pred_prob_df['correct'].value_counts())

        y_pred_prob_df.to_csv(f'{self.save_path}\\poc_results.csv', index_label='game_number')

        print('Results saved')

        