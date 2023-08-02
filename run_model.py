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
from Regression import log_training
import redis 
import os
from urllib.parse import urlparse


class Connection_():
    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL')
        self.parsed_url = urlparse(self.redis_url)
        self.redis_client = redis.Redis(host=self.parsed_url.hostname, port=self.parsed_url.port, password=self.parsed_url.password)

        super().__init__()

    def fetch_current_games(self):


