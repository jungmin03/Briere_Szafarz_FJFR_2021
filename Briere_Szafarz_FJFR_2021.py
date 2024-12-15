import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import pandas_datareader.data as web
import datetime

# 시작 날짜와 끝 날짜 설정
start = datetime.datetime(1926, 1, 1)  # Kenneth R. French 데이터는 1926년부터 제공됩니다.
end = datetime.datetime.today()        # 현재 날짜까지

# 데이터 다운로드
try:
    ff_factors = web.DataReader("F-F_Research_Data_Factors", "famafrench", start, end)
    # 데이터는 여러 테이블로 반환되므로 0번 테이블을 가져옵니다.
    ff_data = ff_factors[0]
    
    print("Fama/French 3 Factors 데이터 다운로드 완료")
    print(ff_data.head())
except Exception as e:
    print(f"데이터를 다운로드하는 데 오류가 발생했습니다: {e}")



# 1. 데이터 로드 및 정리
url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
factors = pd.read_csv(url, compression='zip', skiprows=3, index_col=0)

# 유효한 날짜 형식만 필터링
factors = factors[~factors.index.isna()]
factors = factors[factors.index.astype(str).str.match(r'^\d{6}$')]

# 날짜 형식 변환 (YYYYMM -> datetime)
factors.index = pd.to_datetime(factors.index, format='%Y%m')

# 데이터를 소수점 단위로 변환
factors = factors.apply(pd.to_numeric, errors='coerce') / 100

# 팩터 데이터
factor_returns = factors[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]
risk_free_rate = factors['RF']
excess_factor_returns = factor_returns.add(risk_free_rate, axis=0)  # 초과 수익률 포함

# 섹터 데이터 (가상 데이터 생성)
np.random.seed(42)
sector_returns = pd.DataFrame(
    np.random.normal(0.005, 0.02, (len(factor_returns), 10)),  # 10개 섹터
    index=factor_returns.index,
    columns=[f"Sector_{i}" for i in range(1, 11)]
)

# 데이터 요약
factor_mean = excess_factor_returns.mean()
factor_cov = excess_factor_returns.cov()
sector_mean = sector_returns.mean()
sector_cov = sector_returns.cov()

# 2. 포트폴리오 성과 계산 함수
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, volatility

# 3. 롱숏 포트폴리오 최적화
def long_short_optimization(mean_returns, cov_matrix):
    def neg_sharpe_ratio(weights):
        returns, volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        return -returns / volatility

    num_assets = len(mean_returns)
    bounds = tuple((-1, 1) for _ in range(num_assets))  # 롱숏 허용
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights)}  # 합 = 0

    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

long_short_weights = long_short_optimization(factor_mean, factor_cov)
print("Long-Short Weights (Factors):", long_short_weights)

# 4. 롱온리 포트폴리오 최적화
def long_only_optimization(mean_returns, cov_matrix):
    def neg_sharpe_ratio(weights):
        returns, volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        return -returns / volatility

    num_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(num_assets))  # 롱온리 허용
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # 합 = 1

    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

long_only_weights = long_only_optimization(factor_mean, factor_cov)
print("Long-Only Weights (Factors):", long_only_weights)

# 5. 섹터 롱온리 포트폴리오 최적화
def sector_long_only_optimization(mean_returns, cov_matrix):
    def neg_sharpe_ratio(weights):
        returns, volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        return -returns / volatility

    num_assets = len(mean_returns)
    bounds = tuple((0, 1) for _ in range(num_assets))  # 롱온리 허용
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # 합 = 1

    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(neg_sharpe_ratio, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

sector_long_only_weights = sector_long_only_optimization(sector_mean, sector_cov)
print("Long-Only Weights (Sectors):", sector_long_only_weights)

# 6. 성과 분석
def calculate_sharpe_ratio(portfolio_returns, risk_free_rate):
    excess_returns = portfolio_returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()

# 팩터 및 섹터 포트폴리오 수익률 계산
factor_portfolio_returns = excess_factor_returns @ long_only_weights
sector_portfolio_returns = sector_returns @ sector_long_only_weights

# 샤프 비율 계산
factor_sharpe = calculate_sharpe_ratio(factor_portfolio_returns, risk_free_rate.mean())
sector_sharpe = calculate_sharpe_ratio(sector_portfolio_returns, risk_free_rate.mean())

print(f"Factor Portfolio Sharpe Ratio: {factor_sharpe:.2f}")
print(f"Sector Portfolio Sharpe Ratio: {sector_sharpe:.2f}")

# 7. 효율적 경계 시각화
def efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(mean_returns)))
        weights_record.append(weights)
        returns, volatility = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = returns
        results[1, i] = volatility
        results[2, i] = results[0, i] / results[1, i]  # 샤프 비율

    return results, weights_record

factor_results, _ = efficient_frontier(factor_mean, factor_cov)
sector_results, _ = efficient_frontier(sector_mean, sector_cov)

# 시각화
plt.figure(figsize=(10, 6))
plt.scatter(factor_results[1, :], factor_results[0, :], c=factor_results[2, :], cmap='viridis', label='Factors')
plt.scatter(sector_results[1, :], sector_results[0, :], c=sector_results[2, :], cmap='plasma', label='Sectors')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.title('Efficient Frontier: Factors vs Sectors')
plt.legend()
plt.show()
