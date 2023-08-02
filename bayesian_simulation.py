import pandas as pd

def calculate_bets(df, type_, i):
    # Initialize bankroll
    bankroll = 2500
    df['game_won'] = 0
    df['game_lost'] = 0

    df['win'] = 0
    df['loss'] = 0
    # New columns for bet_amount, bet_win, and updated bankroll
    df['bet_amount'] = 0
    df['bet_win'] = 0
    df['bankroll'] = 0

    win = 0
    loss = 0
    df['win'] = 0
    df['loss'] = 0

    for idx, row in df.iterrows():
        bet = 0
        bet_win = 0
        game_won = 0
        game_lost = 0
        # Calculate bet amount as 5% of current bankroll
        if idx == 0:
            bet = bankroll * 0.05
            segment_br = bankroll
        elif idx % 5 == 0:
            bet = bankroll * 0.05
            segment_br = bankroll
        else:
            bet = segment_br * 0.05

        if row['bethome']:
            if row['home_is_winner'] == 1:
                win_multiplier = (100 / abs(row['home_odds'])) if row['home_odds'] < 0 else (row['home_odds'] / 100)
                bet_win = bet * win_multiplier
                bankroll += bet_win  # Add the winnings to the bankroll
                win += 1
                game_won = 1
                game_lost = 0
            else:
                bankroll -= bet  # Decrease the bankroll by the bet amount
                bet_win = 0
                loss += 1
                game_won = 0
                game_lost = 1

        elif row['betaway']:
            if row['home_is_winner'] == 0:
                win_multiplier = (100 / abs(row['away_odds'])) if row['away_odds'] < 0 else (row['away_odds'] / 100)
                bet_win = bet * win_multiplier
                bankroll += bet_win  # Add the winnings to the bankroll
                win += 1
                game_won = 1
                game_lost = 0
            else:
                bankroll -= bet  # Decrease the bankroll by the bet amount
                bet_win = 0
                loss += 1
                game_won = 0
                game_lost = 1
        else:
            bet = 0 
        # Update the DataFrame
        df.at[idx, 'bet_amount'] = bet
        df.at[idx, 'net_bet_win'] = bet_win
        df.at[idx, 'bankroll'] = bankroll
        df.at[idx, 'game_won'] = game_won
        df.at[idx, 'game_lost'] = game_lost
        df.at[idx, 'total_won_games'] = win
        df.at[idx, 'total_lost_games'] = loss
    
    if df['bankroll'].iloc[-1] > 1000:
        winner = f'bayesian_tests/experiment_21/simulations/{type_}/5_pct/win/simulation_set{i}.csv'
        df.to_csv(winner)
        bankroll = df['bankroll'].iloc[-1]
    else:
        losers = f'bayesian_tests/experiment_21/simulations/{type_}/5_pct/lost/simulation_set{i}.csv'
        df.to_csv(losers)
        bankroll = df['bankroll'].iloc[-1]
    
    return bankroll

def run_simulation():
    types = ['no_calibration', 'isotonic', 'sigmoid']

    for type_ in types:
        bankrolls = {}
        for i in range(1, 100):
            df = pd.read_csv(f'bayesian_tests/experiment_21/predictions/{type_}/test_predictions_{i}.csv')
            df['homediff'] = df['model_home_win_prob'] - df['home_implied_prob'] 
            df['awaydiff'] = df['model_away_win_prob'] - df['away_implied_prob']
            df.drop_duplicates(subset='id', keep='first', inplace=True)

            df['betaway']  = ((df['homediff'] > 0.01))
            df['bethome'] = ((df['awaydiff'] >  0.01))

            bankrolls[i] = calculate_bets(df, type_, i)

        df = pd.DataFrame.from_dict(bankrolls, orient='index', columns=['bankroll_totals'])
        df.to_csv(f'bayesian_tests/experiment_21/simulations/{type_}/best_results.csv')

#run_simulation()