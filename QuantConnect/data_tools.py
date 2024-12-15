from AlgorithmImports import *

class LastDateHandler():
    _last_update_date:Dict[Symbol, datetime.date] = {}

    @staticmethod
    def get_last_update_date() -> Dict[Symbol, datetime.date]:
       return LastDateHandler._last_update_date

# Quantpedia data.
# NOTE: IMPORTANT: Data order must be ascending (datewise)
class QuantpediaFamaFrench(PythonData):
    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(f'data.quantpedia.com/backtesting_data/equity/fama_french/{config.Symbol.Value.lower()}.csv', SubscriptionTransportMedium.RemoteFile, FileFormat.Csv)

    def Reader(self, config, line, date, isLiveMode):
        data = QuantpediaFamaFrench()
        data.Symbol = config.Symbol
        
        if not line[0].isdigit():
            return None
        
        split = line.split(';')
        
        data.Time = datetime.strptime(split[0], "%d.%m.%Y") + timedelta(days=1)
        data['market'] = float(split[1])
        data['size'] = float(split[2])
        data['value'] = float(split[3])
        data['profitability'] = float(split[4])
        data['investment'] = float(split[5])

        if config.Symbol not in LastDateHandler._last_update_date:
            LastDateHandler._last_update_date[config.Symbol] = datetime(1,1,1).date()
        if data.Time.date() > LastDateHandler._last_update_date[config.Symbol]:
            LastDateHandler._last_update_date[config.Symbol] = data.Time.date()

        return data

class QuantpediaFamaFrenchEquity(PythonData):
    def GetSource(self, config, date, isLiveMode):
        return SubscriptionDataSource(f'data.quantpedia.com/backtesting_data/equity/fama_french/{config.Symbol.Value.lower()}.csv', SubscriptionTransportMedium.RemoteFile, FileFormat.Csv)

    def Reader(self, config, line, date, isLiveMode):
        data = QuantpediaFamaFrenchEquity()
        data.Symbol = config.Symbol
        
        if not line[0].isdigit():
            return None
        
        split = line.split(';')
        
        data.Time = datetime.strptime(split[0], "%d.%m.%Y") + timedelta(days=1)
        data.Value = float(split[1])

        if config.Symbol.Value not in LastDateHandler._last_update_date:
            LastDateHandler._last_update_date[config.Symbol.Value] = datetime(1,1,1).date()
        if data.Time.date() > LastDateHandler._last_update_date[config.Symbol.Value]:
            LastDateHandler._last_update_date[config.Symbol.Value] = data.Time.date()

        return data

# custom fee model
class CustomFeeModel:
    def GetOrderFee(self, parameters):
        fee = parameters.Security.Price * parameters.Order.AbsoluteQuantity * 0.00005
        return OrderFee(CashAmount(fee, "USD"))