# statsArb

Several implementations of statistical arbitrage strategies. 

1. Model Returns residuals as an OU process with fixed entry and exits detailed in "Statistical Arbitrage in the U.S. Equities Market" by Marco Avellaneda and Jeong-Hyun Lee 2008
    1. defactoring single stock returns with the same industry ETF (which holds the stock as well) returns
    2. defactoring single stock returns with 15 PCA decomposed market eigen-portfolios
    3. defactoring with 5 factor ETFs (small-cap, value, low volatility, growth and momentum)
2. Model Mean-reversal of mutual information as OU process
3. SVM or XGB classifer of widened spread reversal

