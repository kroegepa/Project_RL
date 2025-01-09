import typing
import pandas as pd

class Actor():
    def __init__ (self):
        pass
    def act(self):
        pass

class UniformBuyActor(Actor):
    def act(self, state) -> float:
        return 0.5

class MovingAverageActor(Actor):
    def __init__(self, path_to_data):
        super().__init__()
        self.df = pd.read_excel(path_to_data)
        self.price_values = self.df.iloc[:, 1:25]

    def act(self, state, window_size_d=7, hour_buffer=1) -> float:
        current_day = state[3] - 1
        current_hour = state[2] - 1
        current_price = state[1]
        days = [current_day - i for i in range(window_size_d)]
        hours = [current_hour + i for i in range(-hour_buffer, hour_buffer+1)]
        print(days)
        print(hours)
        subset = self.price_values.iloc[days, hours]
        moving_average = subset.mean(axis=None)
        # TODO: pick an action based on moving average
        return moving_average, current_price