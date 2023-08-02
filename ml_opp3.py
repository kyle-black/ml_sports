import pandas as pd
import numpy as np
import sqlite3
import odds_pull
from bayesian_approach import bayesian_test 
from bayesian_predictions import run_predictions
from bayesian_simulation import run_simulation
from book_measurement import measurement_
from ensemble_learn2 import XGBoost_train
from book_measurement import measurement_
from Regression import log_training, linear_training
from neural_training import bayesian_neural_network
from new_neural import bayesian_network

class Pull_Data():
    def __init__(self):
        self.connect_name = 'mlb_games.db'
        self.connect = sqlite3.connect(self.connect_name)
        self.past_games = pd.read_sql('SELECT * FROM past_games_update_5', self.connect)
        self.odds = pd.read_sql('SELECT * FROM book_odds12', self.connect)
        self.win_pct = pd.read_csv('date_stats/winpct.csv')
        self.runs = pd.read_csv('date_stats/stats_R.csv')
        self.opp_runs = pd.read_csv('date_stats/stats_opp_R.csv')
        #self.past_games['game_date'] = pd.to_datetime(self.past_games['game_date'])
class Combine_Data(Pull_Data):
    def __init__(self):
        # Call parent's init method
        super().__init__()
        #self.home_win_pct = pd.DataFrame()
        #self.away_win_pct = pd.DataFrame()

    def normalize(self,series):
        return (series - series.min()) / (series.max() - series.min())
    

    def add_rolling_pct(self, df_,type_,days=[10, 30]):
        df_ = df_.sort_values('Date')  # Sort by date
        df_.set_index('Date', inplace=True)  # Set date as index

        for day in days:
            # Group by team and calculate rolling mean for each team
            df_[f'Rolling_{day}D_{type_}'] = df_.groupby('Team')['Current'].transform(
            lambda x: x.rolling(day).mean())

        df_.reset_index(inplace=True)  # Reset index

        return df_



 ############################### Add Win Pct ##################################       
    def win_pct_add(self):
        self.win_pct = self.win_pct[['Date','Team', 'Current','Last 3','Last 1', 'Home', 'Away', 'Previous']]
        self.win_pct = self.win_pct.replace('--', np.nan)
        self.win_pct[['Current','Last 3','Last 1', 'Home', 'Away', 'Previous']]=self.win_pct[['Current','Last 3','Last 1', 'Home', 'Away', 'Previous']].apply(pd.to_numeric)
        
        ##################ADD ROLLING PCT FUNCTION
        self.win_pct = self.add_rolling_pct(self.win_pct,'win')
        #######################################################


        
        
        self.home_win_pct = self.win_pct.copy()
        self.away_win_pct = self.win_pct.copy()

        ###################RENAME COLUMNS 
        self.home_win_pct.rename(columns ={'Date':'home_Date','Team':'home_Team', 'Current':'home_Current_win_pct','Rolling_10D_win':'home_Rolling_10D_win' ,'Rolling_30D_win':'home_Rolling_30D_win','Last 3':'home_3_win_pct','Last 1':'home_1_win_pct','Home':'home_Home_win_pct', 'Away': 'home_Away_win_pct', 'Previous':'home_prev_win_pct'}, inplace =True)
        for col in ['home_Current_win_pct','home_3_win_pct', 'home_1_win_pct', 'home_Home_win_pct', 'home_Away_win_pct', 'home_prev_win_pct','home_Rolling_10D_win', 'home_Rolling_30D_win']:
            self.home_win_pct[col] = self.home_win_pct.groupby('home_Date')[col].transform(self.normalize)
        ################################################################################
        
        
        teamname = {'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago White Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','San Francisco': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
        self.home_win_pct['home_Team'].replace(teamname, inplace =True)
        
        self.away_win_pct.rename(columns ={'Date':'away_Date','Team':'away_Team', 'Current':'away_Current_win_pct', 'Last 3':'away_3_win_pct','Last 1':'away_1_win_pct','Rolling_10D_win':'away_Rolling_10D_win' ,'Rolling_30D_win':'away_Rolling_30D_win','Home':'away_Home_win_pct', 'Away': 'away_Away_win_pct', 'Previous':'away_prev_win_pct'}, inplace=True)
        for col in ['away_Current_win_pct','away_3_win_pct', 'away_1_win_pct', 'away_Home_win_pct', 'away_Away_win_pct', 'away_prev_win_pct','away_Rolling_10D_win', 'away_Rolling_30D_win']:
            self.away_win_pct[col] = self.away_win_pct.groupby('away_Date')[col].transform(self.normalize)

        self.away_win_pct['away_Team'].replace(teamname, inplace =True)

        return self.home_win_pct, self.away_win_pct
    
############################### Add Win Pct ##################################     
    def combine_game_win(self):
        self.home_win_pct, self.away_win_pct = self.win_pct_add()
        self.past_games['game_date'] = pd.to_datetime(self.past_games['game_date']).dt.date
        self.home_win_pct['home_Date'] = pd.to_datetime(self.home_win_pct['home_Date']).dt.date
        self.away_win_pct['away_Date'] = pd.to_datetime(self.away_win_pct['away_Date']).dt.date
        self.merged_df = self.past_games.merge(self.home_win_pct, how='inner',  left_on=['home_team', 'game_date'], right_on = ['home_Team', 'home_Date'])
        self.merged_df = self.merged_df.merge(self.away_win_pct, how='inner', left_on=['away_team', 'game_date'], right_on = ['away_Team', 'away_Date'])

        return self.merged_df
    
    def find_home_cols(self,df):
            return [col for col in df if col.endswith('_home')]

    def find_away_cols(self,df):
            return [col for col in df if col.endswith('_away')]
    
    def combine_game_odds(self):
        self.combine_game_win()
        
         
            
        self.odds = self.odds.groupby('game_id').apply(lambda group: group.fillna(method='ffill').fillna(method='bfill'))
        #self.odds = self.odds.drop_duplicates(subset='game_id', keep='last')
        self.odds['game_date'] = pd.to_datetime(self.odds['commence_time']).dt.tz_convert('US/Eastern').dt.date
        home_cols = self.find_home_cols(self.odds)
        away_cols = self.find_away_cols(self.odds)
        
        # Replace odds that are >400 or <-400 with NaN in home_odds and away_odds columns
        for col in home_cols:
            self.odds[col] = self.odds[col].where(self.odds[col].between(-400, 400), np.nan)
        for col in away_cols:
            self.odds[col] = self.odds[col].where(self.odds[col].between(-400, 400), np.nan)

        self.odds['home_median'] = self.odds[home_cols].apply(lambda row: row.median() if row.count() >= 5 else np.nan, axis=1)
        self.odds['away_median'] = self.odds[away_cols].apply(lambda row: row.median() if row.count() >= 5 else np.nan, axis=1)
        self.odds['home_mean'] = self.odds[home_cols].apply(lambda row: row.mean() if row.count() >= 5 else np.nan, axis=1)
        self.odds['away_mean'] = self.odds[away_cols].apply(lambda row: row.mean() if row.count() >= 5 else np.nan, axis=1)
        
        self.odds = self.odds.drop_duplicates(subset='game_id', keep='last')
        self.merged_df = self.merged_df.merge(self.odds, how='inner', left_on=['home_team', 'away_team','game_date'], right_on=['home_team','away_team','game_date'])
        
        return self.merged_df
    
    ######################################COMBINE GAME RUNS #########################################
        ######################################COMBINE GAME RUNS #########################################
        
    def combine_game_runs(self):
        self.merged_df =self.combine_game_odds()
        self.runs.drop(['Unnamed: 0','Unnamed: 9'], inplace=True, axis =1)
        #self.win_pct = self.win_pct[['Date','Team', 'Current','Last 3','Last 1', 'Home', 'Away', 'Previous']]

        teamname ={'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','SF Giants': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
        self.runs['Team'].replace(teamname, inplace =True)
        self.runs = self.runs.replace('--', np.nan)
        self.runs[['Current','Last 3','Last 1', 'Home', 'Away', 'Previous']]=self.win_pct[['Current','Last 3','Last 1', 'Home', 'Away', 'Previous']].apply(pd.to_numeric)
        
        ##################ADD ROLLING PCT FUNCTION
        self.runs = self.add_rolling_pct(self.runs,'runs')
        #######################################################
        self.home_runs = self.runs.copy()
        self.away_runs = self.runs.copy()
        ######################################################
        ###################RENAME COLUMNS 
        self.home_runs.rename(columns ={'Date':'home_Date','Team':'home_Team', 'Current':'home_r_Current', 'Last 3':'home_r_3','Last 1':'home_r_1','Home':'home_r_Home', 'Away': 'home_r_Away', 'Previous':'home_r_prev','Rolling_10D_runs':'Rolling_10D_r_home','Rolling_30D_runs':'Rolling_30D_r_home'}, inplace =True)
        for col in ['home_r_Current','home_r_3', 'home_r_1', 'home_r_Home', 'home_r_Away', 'home_r_prev','Rolling_10D_r_home', 'Rolling_30D_r_home']:
            self.home_runs[col] = self.home_runs.groupby('home_Date')[col].transform(self.normalize)
        ################################################################################

         ###################RENAME COLUMNS 
        self.away_runs.rename(columns ={'Date':'away_Date','Team':'away_Team', 'Current':'away_r_Current', 'Last 3':'away_r_3','Last 1':'away_r_1','Home':'away_r_Home', 'Away': 'away_r_Away', 'Previous':'away_r_prev','Rolling_10D_runs':'Rolling_10D_r_away','Rolling_30D_runs':'Rolling_30D_r_away'}, inplace =True)
        #self.away_runs.rename(columns ={'Date':'away_Date','Team':'away_Team', 'Current':'away_r_Current', 'Last 3':'away_r_3','Last 1':'away_r_1','Home':'away_r_Home', 'Away': 'away_r_Away', 'Previous':'away_r_prev'}, inplace =True)
        #return self.away_runs
        
        for col in ['away_r_Current','away_r_3', 'away_r_1', 'away_r_Home', 'away_r_Away', 'away_r_prev','Rolling_10D_r_away', 'Rolling_30D_r_away']:
            self.away_runs[col] = self.away_runs.groupby('away_Date')[col].transform(self.normalize)
        ################################################################################
        
        self.home_runs['home_r_Date'] = pd.to_datetime(self.home_runs['home_Date'], errors='coerce').dt.date
        self.away_runs['away_r_Date'] = pd.to_datetime(self.away_runs['away_Date'], errors='coerce').dt.date

        self.merged_df=self.merged_df.merge(self.home_runs, how='inner', left_on=['home_team','game_date'], right_on=['home_Team','home_r_Date'])
        self.merged_df=self.merged_df.merge(self.away_runs, how='inner', left_on=['away_team','game_date'], right_on=['away_Team','away_r_Date'])


        return self.merged_df
       
######################################COMBINE GAME OPP RUNS #########################################
    
    def combine_game_runs_opp(self):
            self.merged_df =self.combine_game_runs()
           # self.runs.drop(['Unnamed: 0','Unnamed: 9'], inplace=True, axis =1)
            #self.win_pct = self.win_pct[['Date','Team', 'Current','Last 3','Last 1', 'Home', 'Away', 'Previous']]

            teamname ={'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','SF Giants': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
            self.opp_runs['Team'].replace(teamname, inplace =True)
            self.opp_runs = self.runs.replace('--', np.nan)
            self.opp_runs[['Current','Last 3','Last 1', 'Home', 'Away', 'Previous']]=self.win_pct[['Current','Last 3','Last 1', 'Home', 'Away', 'Previous']].apply(pd.to_numeric)
            
            ##################ADD ROLLING PCT FUNCTION
            self.opp_runs = self.add_rolling_pct(self.opp_runs,'opp_runs')
            #######################################################
            self.home_opp_runs = self.opp_runs.copy()
            self.away_opp_runs = self.opp_runs.copy()
            ######################################################
            ###################RENAME COLUMNS 
            self.home_opp_runs.rename(columns ={'Date':'home_Date','Team':'home_Team', 'Current':'home_opp_Current', 'Last 3':'home_opp_3','Last 1':'home_opp_1','Home':'home_opp_Home', 'Away': 'home_opp_Away', 'Previous':'home_opp_prev','Rolling_10D_opp_runs':'Rolling_10D_opp_home','Rolling_30D_opp_runs':'Rolling_30D_opp_home'}, inplace =True)
            for col in ['home_opp_Current','home_opp_3', 'home_opp_1', 'home_opp_Home', 'home_opp_Away', 'home_opp_prev','Rolling_10D_opp_home', 'Rolling_30D_opp_home']:
                self.home_opp_runs[col] = self.home_opp_runs.groupby('home_Date')[col].transform(self.normalize)
            ################################################################################

            ###################RENAME COLUMNS 
            self.away_opp_runs.rename(columns ={'Date':'away_Date','Team':'away_Team', 'Current':'away_opp_Current', 'Last 3':'away_opp_3','Last 1':'away_opp_1','Home':'away_opp_Home', 'Away': 'away_opp_Away', 'Previous':'away_opp_prev','Rolling_10D_opp_runs':'Rolling_10D_opp_away','Rolling_30D_opp_runs':'Rolling_30D_opp_away'}, inplace =True)
            for col in ['away_opp_Current','away_opp_3', 'away_opp_1', 'away_opp_Home', 'away_opp_Away', 'away_opp_prev','Rolling_10D_opp_away', 'Rolling_30D_opp_away']:
                self.away_opp_runs[col] = self.away_opp_runs.groupby('away_Date')[col].transform(self.normalize)
            ################################################################################
            
            self.home_opp_runs['home_opp_Date'] = pd.to_datetime(self.home_opp_runs['home_Date'], errors='coerce').dt.date
            self.away_opp_runs['away_opp_Date'] = pd.to_datetime(self.away_opp_runs['away_Date'], errors='coerce').dt.date

            self.merged_df=self.merged_df.merge(self.home_opp_runs, how='inner', left_on=['home_team','game_date'], right_on=['home_Team','home_opp_Date'])
            self.merged_df=self.merged_df.merge(self.away_opp_runs, how='inner', left_on=['away_team','game_date'], right_on=['away_Team','away_opp_Date'])


            return self.merged_df
   
        

    

    def american_to_implied_probability(self,american_odds):
        """
        Convert American odds to implied probability
        """
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)

    def calculate_vig(self,row):
        """
        Calculate the vig given two American odds
        """
        # Calculate the implied probabilities from the odds
        prob1 = self.american_to_implied_probability(row['lowvig_home'])
        prob2 = self.american_to_implied_probability(row['lowvig_away'])

        # The vig is the excess of these probabilities over 1
        vig = prob1 + prob2 - 1
        return vig
    

    def update_columns(self):
        #combine_game_runs_opp
        self.merged_df = self.combine_game_runs_opp()

        
        
       # self.merged_df = self.merged_df.replace('--', np.nan).dropna()
        self.merged_df = self.merged_df.drop_duplicates(subset='id', keep='last')

        self.home_cols = [col for col in self.merged_df.columns if col.endswith('_home')]
        self.away_cols = [col for col in self.merged_df.columns if col.endswith('_away')]
        self.home_vf_cols = []
        self.away_vf_cols = []
# Loop through the columns and apply the calculations
        for home_col, away_col in zip(self.home_cols, self.away_cols):
            home_prob_col = f'{home_col}_prob'
            away_prob_col = f'{away_col}_prob'
            self.home_vf_col = f'{home_col}_vf'
            self.away_vf_col = f'{away_col}_vf'

            self.merged_df[home_prob_col] = self.merged_df[home_col].apply(self.american_to_implied_probability)
            self.merged_df[away_prob_col] = self.merged_df[away_col].apply(self.american_to_implied_probability)
    
    # Calculate vig free probabilities
            total_prob = self.merged_df[home_prob_col] + self.merged_df[away_prob_col]
            self.merged_df[self.home_vf_col] = self.merged_df[home_prob_col] / total_prob
            self.merged_df[self.away_vf_col] = self.merged_df[away_prob_col] / total_prob

            self.home_vf_cols.append(self.home_vf_col)
            self.away_vf_cols.append(self.away_vf_col)

        # For columns ending with '_home_vf'
        home_vf_cols = [col for col in self.merged_df.columns if '_home_vf' in col]
        self.merged_df['max_min_diff_home_vf'] = self.merged_df[home_vf_cols].max(axis=1) - self.merged_df[home_vf_cols].min(axis=1)

        # For columns ending with '_away_vf'
        away_vf_cols = [col for col in self.merged_df.columns if '_away_vf' in col]
        self.merged_df['max_min_diff_away_vf'] = self.merged_df[away_vf_cols].max(axis=1) - self.merged_df[away_vf_cols].min(axis=1)

        #self.merged_df[]
        
        return self.merged_df
    

class Train_Model(Combine_Data):
    def __init__(self):
        # Call parent's init method
        super().__init__()
        self.update_columns()
        # Initialize attributes


   
        
    
    def prepare_data(self):
        self.test_columns = self.merged_df.columns
        self.merged_df =self.merged_df[self.merged_df['home_is_winner'] !='Unknown']
        self.merged_df = self.merged_df[self.merged_df['game_type'] =='R']

       
        
        
        self.merged_df=self.merged_df[['home_is_winner','lowvig_home_vf','lowvig_away_vf',
                                       'home_Current_win_pct','home_prev_win_pct','home_Rolling_10D_win','home_Rolling_30D_win', 'away_Current_win_pct', 'away_prev_win_pct', 
         'away_Rolling_10D_win', 'away_Rolling_30D_win', 'home_r_Current','home_r_3', 'home_r_1', 'home_r_Home', 'home_r_Away', 'home_r_prev','Rolling_10D_r_home', 'Rolling_30D_r_home', 'away_r_Current','away_r_3', 'away_r_1', 
         'away_r_Home', 'away_r_Away', 'away_r_prev','Rolling_10D_r_away', 'Rolling_30D_r_away', 'home_opp_Current','home_opp_3', 'home_opp_1', 'home_opp_Home', 'home_opp_Away', 'home_opp_prev','Rolling_10D_opp_home', 
         'Rolling_30D_opp_home','away_opp_Current','away_opp_3', 'away_opp_1', 'away_opp_Home', 'away_opp_Away', 'away_opp_prev','Rolling_10D_opp_away', 'Rolling_30D_opp_away'  ]]
        self.merged_df.dropna(inplace=True)
        self.y = self.merged_df['home_is_winner'].astype(int)
        self.X = self.merged_df.drop('home_is_winner', axis=1)
        self.train_cols = self.X.columns
        #return selself.merged_df
        #return self.merged_df
        return self.X,self.y,self.train_cols,self.merged_df
    
    
    def run_training(self):
        self.X, self.y, self.train_cols, self.merged_df = self.prepare_data()
        #df= bayesian_test(self.train_cols,self.X,self.y,self.merged_df)
       # df = log_training(self.X, self.y, self.train_cols)
        df =bayesian_network(self.merged_df,self.train_cols)
       # df = bayesian_neural_network(self.X, self.y, self.train_cols)
        #df = linear_training(self.X, self.y, self.train_cols)
        #df = XGBoost_train(self.merged_df, self.train_cols)
        #df = XGBoost_train(self.X, self.y,self.train_cols, self.merged_df)
        #df.to_csv('test_df12.csv')
        
        
        return df
        #return self.merged_df
    def predict_(self):
        self.run_training()
        self.test_model =  'bayesian_tests/experiment_20/test_data'

        self.test_columns

        run_predictions(self.train_cols,self.test_model)
    
    def simulation_(self):
        self.predict_()
        run_simulation()

       # return df



class Measure(Combine_Data):
    def __init__(self):
        super().__init__()
        self.merged_df =self.update_columns()

    def measure_books(self):
        return measurement_(self.merged_df)

        

#d = Combine_Data()

#print(d.prepare_data() )


d= Train_Model()
print(d.run_training())

    

#d =Combine_Data()
#print(d.update_columns())
#x =d.combine_game_odds()
#x.to_csv('data_checker2.csv')