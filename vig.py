def american_to_implied_probability(american_odds):
    """
    Convert American odds to implied probability
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)
    

def calculate_vig(american_odds1, american_odds2):
    """
    Calculate the vig given two American odds
    """

    # Calculate the implied probabilities from the odds
    prob1 = american_to_implied_probability(american_odds1)
    prob2 = american_to_implied_probability(american_odds2)

    # The vig is the excess of these probabilities over 1
    vig = prob1 + prob2 - 1

    return vig


home_odds = 118
away_odds = -139

vig = calculate_vig(home_odds, away_odds)
print(f"The vig is {vig}")