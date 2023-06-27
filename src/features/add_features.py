class FeatureGenerator:
    def __init__(self, df):
        """Initializes the FeatureGenerator with the DataFrame to process."""
        self.df = df.copy()
    
    def true_shooting_percentage(self):
        """Calculates True Shooting Percentage and adds it to the DataFrame."""
        self.df['TSA'] = self.df['FGA'] + (0.44*self.df['FTA'])
        self.df['TS%'] = self.df['PTS']/(2*self.df['TSA'])
        return self

    def three_point_net_gain(self):
        """Calculates 3 Point Net Gain and adds it to the DataFrame."""
        self.df['3NG'] = (self.df['3P Made'] * 1.94) - ((self.df['3PA'] - self.df['3P Made'])*1.06)
        return self
    
    def turnover_percentage(self):
        """Calculates Turnover Percentage and adds it to the DataFrame."""
        self.df['TOV%'] = (self.df['TOV'] * 100)/(self.df['TSA'] + self.df['TOV'])
        return self
    
    def extra_possession(self):
        """Calculates Extra Possession and adds it to the DataFrame."""
        self.df['Poss_Added'] = self.df['STL'] + self.df['BLK'] + self.df['OREB']
        return self
    
    def generate(self):
        """Generates all the features by calling each method in turn."""
        self.true_shooting_percentage()
        self.three_point_net_gain()
        self.turnover_percentage()
        self.extra_possession()
        return self.df

