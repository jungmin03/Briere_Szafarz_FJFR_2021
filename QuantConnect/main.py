# https://quantpedia.com/strategies/mean-variance-factor-timing/
#
# The investment universe consists of all AMEX, NYSE, and NASDAQ-listed U.S. stocks. The data come from Kenneth French’s website. Create factor portfolios based on five factors: 
# size, value, momentum, investment, and profitability.
# Using the Markowitz model, construct a long-short efficient portfolio maximizing the Sharpe ratio. Each month run out-of-sample estimation using previous 60-month data.
#
# QC Implementation changes:

#region imports
from AlgorithmImports import *
from scipy.optimize import minimize
import data_tools
#endregion

class MeanVarianceFactorTiming(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2010, 1, 1)
        self.SetCash(100000)
        
        self.period:int = 60 * 21

        # warm up fama french values for idiosyncratic volatility
        self.SetWarmup(self.period, Resolution.Daily)

        self.data:dict = {}
        
        self.fama_french_symbol:Symbol = self.AddData(data_tools.QuantpediaFamaFrench, 'fama_french_5_factor', Resolution.Daily).Symbol
        self.ff_factor_names:list[str] = ['market', 'size', 'value', 'profitability', 'investment']

        # ff performance data
        self.fama_french_data:dict = { ff_factor_name : RollingWindow[float](self.period) for ff_factor_name in self.ff_factor_names }
        
        # ff traded symbols
        for factor_name in self.ff_factor_names:
            data:Security = self.AddData(data_tools.QuantpediaFamaFrenchEquity, f'fama_french_5_{factor_name}_eq', Resolution.Daily)
            data.SetLeverage(3)
            data.SetFeeModel(data_tools.CustomFeeModel())

        self.recent_month:int = -1
        self.settings.minimum_order_margin_portfolio_percentage = 0.

    def OnData(self, data):
        # Check if custom data is still coming.
        if any(
            [
            self.securities[x].get_last_data() and self.time.date() > data_tools.LastDateHandler.get_last_update_date()[x] 
            for x in [(f'fama_french_5_{factor_name}_eq').upper() for factor_name in self.ff_factor_names] + [self.fama_french_symbol]
            ]
        ):
            self.liquidate()
            return

        # update fama french values on daily basis
        if self.fama_french_symbol in data and data[self.fama_french_symbol]:
            for ff_factor_name in self.ff_factor_names:
                self.fama_french_data[ff_factor_name].Add(data[self.fama_french_symbol].GetProperty(ff_factor_name))
        
        if self.recent_month == self.Time.month:
            return
        self.recent_month = self.Time.month

        # optimization
        if all(x[1].IsReady for x in self.fama_french_data.items()):
            perf_df:pd.DataFrame = pd.DataFrame(columns=self.ff_factor_names)
            for ff_factor_name in self.ff_factor_names:
                perf_df[ff_factor_name] = np.array([x for x in self.fama_french_data[ff_factor_name]][::-1])

            opt, weights = self.optimization_method(perf_df)
            for ff_factor_symbol, w in weights.items():
                traded_symbol:str = f'fama_french_5_{ff_factor_symbol}_eq'
                if abs(w) > 0.001:
                    self.SetHoldings(traded_symbol, w)
                else:
                    self.Liquidate(traded_symbol)
        
    def optimization_method(self, returns:pd.DataFrame):
        '''Maximize sharpe ratio method'''
        # objective function
        fun = lambda weights: - np.sum(returns.mean() * weights) * 252 / np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

        # Constraint #1: The weights can be negative, which means investors can short a security.
        constraints = [{'type': 'eq', 'fun': lambda w: 1 - np.sum(w)}]

        size = returns.columns.size
        x0 = np.array(size * [1. / size])
        # bounds = tuple((self.minimum_weight, self.maximum_weight) for x in range(size))
        bounds = tuple((0, 1) for x in range(size))

        opt = minimize(fun,                         # Objective function
                       x0,                          # Initial guess
                       method='SLSQP',              # Optimization method:  Sequential Least SQuares Programming
                       bounds = bounds,             # Bounds for variables 
                       constraints = constraints)   # Constraints definition

        return opt, pd.Series(opt['x'], index = returns.columns)