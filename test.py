import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import quandl
import pandas_datareader.data as web
from pandas_datareader.famafrench import get_available_datasets
from pandas_datareader import data as pdr
import datetime

# K.Frech's data library
datasets = get_available_datasets()

for item in datasets:
    print(item)


# set up parameter
start = datetime.datetime(1963, 7, 1)  # Kenneth R. French 데이터는 1926년부터 제공됩니다.
end = datetime.datetime(2014, 12, 1)        # 현재 날짜까지

# 1. 데이터 로드 및 정리
# 1-1. facator data
# 1. 데이터 로드 및 정리
ff_factors_dic = web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start, end)
ff_factors_df = pd.DataFrame(ff_factors_dic[0])
ff_factors_df = ff_factors_df[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
ff_factors_df

# 유효한 날짜 형식만 필터링
ff_factors_df = ff_factors_df[~ff_factors_df.index.isna()]

# 날짜 형식을 datetime으로 변환
ff_factors_df.index = ff_factors_df.index.to_timestamp()

# 데이터를 소수점 단위로 변환
ff_factors_df = ff_factors_df.apply(pd.to_numeric, errors='coerce') / 100


# 1-2.Sector data
ff_sectors_dic = web.DataReader("10_Industry_Portfolios", "famafrench", start, end)
ff_sectors_df = pd.DataFrame(ff_sectors_dic[0])
ff_sectors_df

# 유효한 날짜 형식만 필터링
ff_sectors_df = ff_sectors_df[~ff_sectors_df.index.isna()]

# 날짜 형식을 datetime으로 변환
ff_sectors_df.index = ff_sectors_df.index.to_timestamp()

# 데이터를 소수점 단위로 변환
ff_sectors_df = ff_sectors_df.apply(pd.to_numeric, errors='coerce') / 100


# 1-3.무위험 금리 데이터 
risk_free_rate = ff_factors_dic[0]['RF']
risk_free_rate 
risk_free_rate = risk_free_rate / 100  # 퍼센트를 소수로 변환
# 데이터 확인
print(risk_free_rate.head())


# 2.팩터 기반 포트폴리오 구성
# 팩터 기반 포트폴리오 수익률 계산
factor_portfolio = (ff_factors_df["SMB"] + ff_factors_df["HML"] + ff_factors_df["RMW"] + ff_factors_df["CMA"]) / 4
factor_portfolio = factor_portfolio.cumsum()
factor_portfolio

# 결과 확인
plt.figure(figsize=(10, 6))
plt.plot(factor_portfolio, label="Factor Portfolio (Cumulative)")
plt.title("Factor Portfolio Cumulative Returns")
plt.legend()
plt.grid()
plt.show()


# 3.섹터 기반 포트폴리오 구성
# 섹터 기반 포트폴리오 계산 (동일 가중치)
sector_portfolio = ff_sectors_df.mean(axis=1)  # 동일 가중치 평균

# 누적 수익률 계산
sector_portfolio_cumulative = (1 + sector_portfolio).cumprod()

# 결과 확인
plt.figure(figsize=(10, 6))
plt.plot(sector_portfolio_cumulative, label="Sector Portfolio (Cumulative)")
plt.title("Sector Portfolio Cumulative Returns")
plt.legend()
plt.grid()
plt.show()

# 4.시장상황별 분석
# 시장 지수 데이터 추가 (예: Fama-French의 Mkt-RF)
market_returns = ff_factors_df["Mkt-RF"]

# 좋은 시기와 나쁜 시기 정의 (예: 시장 수익률을 기준으로)
good_times = market_returns > market_returns.median()
bad_times = market_returns <= market_returns.median()

# 각 시기에서 팩터 및 섹터 포트폴리오 성과 계산
good_factor_returns = factor_portfolio[good_times].mean()
bad_factor_returns = factor_portfolio[bad_times].mean()

good_sector_returns = sector_portfolio[good_times].mean()
bad_sector_returns = sector_portfolio[bad_times].mean()

# 결과 출력
print(f"Factor Portfolio - Good Times: {good_factor_returns:.4f}, Bad Times: {bad_factor_returns:.4f}")
print(f"Sector Portfolio - Good Times: {good_sector_returns:.4f}, Bad Times: {bad_sector_returns:.4f}")


# 5.성과 지표 계산
# 샤프 비율 계산
def performance_metrics(portfolio, risk_free):
    """
    포트폴리오 성과 지표를 계산하는 함수
    Args:
        portfolio (pd.Series): 포트폴리오 수익률
        risk_free (pd.Series): 무위험 금리
    Returns:
        dict: 수익률, 변동성, 샤프 비율, 최대 손실
    """
    # 초과 수익률
    excess_return = portfolio - risk_free

    # 지표 계산
    mean_return = portfolio.mean() * 12  # 연환산
    volatility = portfolio.std() * np.sqrt(12)  # 연환산
    sharpe_ratio = excess_return.mean() / portfolio.std()
    
    # 최대 손실 계산
    cumulative_returns = (1 + portfolio).cumprod()
    drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
    max_drawdown = drawdown.min()

    return {
        "Return": mean_return,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown
    }



# 6.시각화 및 보고서 생성
plt.figure(figsize=(12, 6))
plt.plot(factor_portfolio, label="Factor Portfolio")
plt.plot(sector_portfolio_cumulative, label="Sector Portfolio")
plt.title("Cumulative Returns: Factor vs Sector Portfolio")
plt.legend()
plt.grid()
plt.show()

# Deacriptive Statics, Sectors and Factors
# 팩터 포트폴리오 성과
factor_metrics = performance_metrics(factor_portfolio, risk_free_rate)

# 섹터 포트폴리오 성과
sector_metrics = performance_metrics(sector_portfolio, risk_free_rate)

# 결과 출력
results = pd.DataFrame([factor_metrics, sector_metrics], index=["Factor Portfolio", "Sector Portfolio"])
print(results)

