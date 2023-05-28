from data import ModelData
from model import TotalScoreRF

def main():
    # Define file and folder paths
    master_path = f'C:\\Users\\sjmee\\Documents\\OneDrive\\Documents\\Projects\\afl_score_prediction\\data'
    model_path = f'C:\\Users\\sjmee\\Documents\\OneDrive\\Documents\\Projects\\afl_score_prediction\\models'

    games_file = f'{master_path}\\games.csv'
    players_file = f'{master_path}\\players.csv'
    stats_file = f'{master_path}\\stats.csv'

    file_paths = [games_file, players_file, stats_file]

    file_name = 'totalScore_model.pickle'

    # Model parameters
    target_var = 'totalScore_buckets'

    # Define the edges of the score buckets
    bins = [0, 120, 150, 180, 210, 240, 300]

    # Run data transformation methods
    data_prep = ModelData(file_paths, bins)

    model_df = data_prep.build_dataset()

    print('Model data preview')

    print(model_df.head())

    # Build and evaluate model and adjusted model 
    model = TotalScoreRF(model_df, target_var, bins, model_path, file_name)

    model.grid_search_train()

    model.make_base_predictions()

    model.make_adj_predictions()
    
    print('Completed successfully!')

if __name__ == '__main__':
    main()