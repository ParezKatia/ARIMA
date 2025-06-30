from pmdarima import auto_arima
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import shapiro

series = well_series  # your W1 data

model = auto_arima(series, seasonal=True, m=12, trace=True,
                   error_action='ignore', suppress_warnings=True, stepwise=True)

print(model.summary())

residuals = model.resid()
print("Shapiro–Wilk:", shapiro(residuals))
print("Ljung–Box:", acorr_ljungbox(residuals, lags=[12]))

plot_acf(residuals)
plt.title("ACF of Residuals")
plt.show()

plot_pacf(residuals)
plt.title("PACF of Residuals")
plt.show()
