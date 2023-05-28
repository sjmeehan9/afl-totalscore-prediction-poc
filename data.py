import pandas as pd
import numpy as np
import datetime

class ModelData:

    def __init__(self, file_paths: list, bins: list):
        self.__file_paths = file_paths
        self.bins = bins

    @property
    def file_paths(self):
        return self.__file_paths

    def __games_data(self):
        # Games data preprocessing
        games_df = pd.read_csv(self.__file_paths[0])

        games_df['date'] = pd.to_datetime(games_df['date'], format='%d-%b-%Y')

        games_df.loc[:,'day_of_week'] = pd.to_datetime(games_df['date']).dt.strftime('%A')

        # Create a mapping of unique day_of_week names to numerical IDs
        weekday_to_id = dict(zip(games_df['day_of_week'].unique(), range(len(games_df['day_of_week'].unique()))))

        # Create a new column with numerical IDs for the venues
        games_df['day_of_week_id'] = games_df['day_of_week'].map(weekday_to_id)

        round_to_id = dict(zip(games_df['round'].unique(), range(len(games_df['round'].unique()))))

        games_df['round_id'] = games_df['round'].map(round_to_id)

        games_df['month'] = games_df['date'].dt.month

        games_df['rain_flag'] = games_df['rainfall'].apply(
            lambda x: 0 if x==0 else (1 if x<=1 else (2 if x<=10 else 3))).astype(int)

        games_df['totalScore'] = games_df['homeTeamScore'] + games_df['awayTeamScore']

        bins = self.bins

        # Define the labels for the buckets
        labels = np.arange(1, len(bins))

        # Use pd.cut() to create the buckets
        games_df['totalScore_buckets'] = pd.cut(games_df['totalScore'], bins=bins, labels=labels)

        games_df['startTimeHour'] = games_df['startTime'].apply(lambda x: datetime.datetime.strptime(x, '%I:%M %p').time().strftime('%H')).astype(int)

        venue_to_id = dict(zip(games_df['venue'].unique(), range(len(games_df['venue'].unique()))))

        games_df['venue_id'] = games_df['venue'].map(venue_to_id)

        return games_df

    def __players_data(self):
        players_df = pd.read_csv(self.__file_paths[1])

        # Split "position" column into two columns
        players_df[['position_1', 'position_2']] = players_df['position'].str.split(', ', expand=True)

        # Filter rows with a non-"None" value in the "position_2" column
        filtered_players_df = players_df[players_df['position_2'].notnull()]

        # Realign "position_2" to "position_1" column
        filtered_players_df = filtered_players_df.drop(columns='position_1')
        filtered_players_df = filtered_players_df.rename(columns={'position_2': 'position_1'})

        # Concatenate the original DataFrame with the new DataFrame
        players_df = players_df.drop(columns='position_2')
        players_df = pd.concat([players_df, filtered_players_df])

        return players_df

    def __stats_data(self):
        games_df = self.__games_data()

        players_df = self.__players_data()

        stats_df = pd.read_csv(self.__file_paths[2])

        # Merge date and day of week into stats_df
        temp_games_df = games_df[['gameId', 'date']]
        stats_df = stats_df.merge(temp_games_df, how='left', on='gameId')

        # Merge DOB and position columns from players_df into stats_df, on playerid
        temp_players_df = players_df[['playerId', 'dob', 'position_1']]
        temp_players_df.loc[:, 'dob'] = pd.to_datetime(temp_players_df['dob'], format='%d-%b-%Y').dt.date
        stats_df = stats_df.merge(temp_players_df, how='left', on='playerId')

        # Calculate player age
        stats_df['age'] = (stats_df['date'] - pd.to_datetime(stats_df['dob'])).dt.days / 365.25
        stats_df['age'] = stats_df['age'].round().astype(int)

        # Calculate average age by team, game and position
        age_by_pos = stats_df.groupby(['gameId', 'team', 'position_1'])['age'].median().reset_index()
        age_by_pos = age_by_pos.pivot_table(index=['gameId', 'team'], columns='position_1', values='age').reset_index()
        age_by_pos = age_by_pos.drop(columns=['Ruck', 'Midfield'])
        age_by_pos['Defender'] = age_by_pos['Defender'].round().astype(int)
        age_by_pos['Forward'] = age_by_pos['Forward'].round().astype(int)
        
        return age_by_pos

    def build_dataset(self):
        games_df = self.__games_data()

        age_by_pos = self.__stats_data()

        # Merge for each game and team into games_df
        games_df = games_df.merge(age_by_pos, how='left', left_on=['gameId', 'homeTeam'], right_on=['gameId', 'team'])
        games_df = games_df.rename(columns={'Defender': 'home_def_age', 
            'Forward': 'home_for_age'})
        games_df = games_df.drop(columns='team')

        games_df = games_df.merge(age_by_pos, how='left', left_on=['gameId', 'awayTeam'], right_on=['gameId', 'team'])
        games_df = games_df.rename(columns={'Defender': 'away_def_age', 
            'Forward': 'away_for_age'})
        games_df = games_df.drop(columns='team')

        # Count the occurrences of each venue_id
        venue_counts = games_df['venue_id'].value_counts()

        # Create a boolean mask that filters for venue_id counts greater than 5
        mask = games_df['venue_id'].map(venue_counts) > 50

        # Filter the DataFrame using the boolean mask
        filtered_model_df = games_df[mask]

        # Create model dataset
        model_df = filtered_model_df[['round_id', 'day_of_week_id', 'month', 'rain_flag', 'totalScore_buckets', 'startTimeHour', 'venue_id'
            , 'home_def_age', 'home_for_age', 'away_def_age', 'away_for_age']]

        return model_df
