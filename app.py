import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from PIL import Image
from autots import AutoTS
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
from darts.utils.statistics import check_seasonality
from darts import TimeSeries
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Start of execution time calculation
start = time.time()

# Setting the app layout
st.set_page_config(layout="wide")

# Display FLNT logo
image = Image.open("flnt.png")
st.sidebar.image(image,
                 width=230)

# Display file uploader (adding space beneath the FLNT logo)
st.sidebar.write("")
st.sidebar.write("")
data1 = st.sidebar.file_uploader("",
                                 type=["csv", "xls", "xlsx"])

st.sidebar.write("---")

# Check for errors during upload
try:
    # Read dataset file uploader
    if data1 is not None:
        if data1.name.endswith(".csv"):
            data_df1 = pd.read_csv(data1)
        else:
            data_df1 = pd.read_excel(data1)

    # Impute missing values with mean
    data_df1 = data_df1.fillna(data_df1.mean())

    # For choosing features and targets
    data_df1_types = data_df1.dtypes.to_dict()

    # Choosing features and target for file 1
    targets1 = []
    for key, val in data_df1_types.items():
        if val != object:
            targets1.append(key)

    help_dependent = "Target variable is the effect. It is the value that you are trying to forecast"
    help_independent = "Explanatory variable is the cause. It is the value which may contribute to the forecast"
    chosen_target1 = st.sidebar.selectbox("Choose target variable",
                                          targets1,
                                          help=help_dependent)
    features1 = list(data_df1_types.keys())
    features1.remove(chosen_target1)
    chosen_date1 = st.sidebar.selectbox("Choose date column to use",
                                        features1)
    chosen_features1 = st.sidebar.multiselect("Choose explanatory variable(s) to use",
                                              features1,
                                              help=help_independent)

    # Create a dataframe based on chosen variables
    new_cols1 = chosen_features1.copy()
    new_cols1.append(chosen_target1)

    data_df1 = data_df1[new_cols1]

    # Preprocess data for experiment setup
    data_df1_series = data_df1.copy()

    # For descriptive stats
    data_df1_cols = data_df1.columns
    data_df1_shape = data_df1.shape

    data_df1_series[chosen_date1] = pd.to_datetime(data_df1_series[chosen_date1],
                                                   dayfirst=True)

    # data_df1_series with date column as datetime format (for darts time series input as dataframe)
    data_df1_series_dt = data_df1_series.copy()

    data_df1_series.set_index(data_df1_series[chosen_date1],
                              inplace=True)
    data_df1_series.drop(chosen_date1,
                         axis=1,
                         inplace=True)

    # Create tabs for plots and statistics
    plot_tab, stat_tab, correlation_tab, forecast_tab = st.tabs(["Plots",
                                                                 "Statistics",
                                                                 "Correlation",
                                                                 "Forecast"])

    # Test for stationarity of time series data
    def test_stationarity(timeseries):
        """
        Performs Augmented Dickey-Fuller test to check stationarity of input time series data

        :param timeseries: time series data to test for stationarity
        :return: boolean True(stationary) or False(non-stationary)
        """
        # perform Dickey-Fuller test
        dftest = adfuller(timeseries,
                          autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic',
                                    'p-value',
                                    '#Lags Used',
                                    'Number of Observations Used'])

        if dfoutput['p-value'] > 0.05:
            return False
        else:
            return True

    # Decompose time series data to get components (trend, seasonal, residual)
    def decompose(df, target_col):
        """
        Performs seasonal decomposition to extract trend, seasonal component, and residual from time series data

        :param df: input time series dataframe
        :param target_col: target column of dataframe
        :return: seasonal component, trend component, residual component (in this order)
        """
        dec = seasonal_decompose(df[target_col])
        return dec.seasonal, dec.trend, dec.resid

    # Test for seasonality
    @st.cache()
    def seasonality_test(
            df: pd.DataFrame,
            time_col: str,
            value_col: str):
        """
        Test for seasonality using ACF

        :param df: pandas dataframe
        :param time_col: time column of dataframe
        :param value_col: target column of dataframe to test for seasonality
        :return: boolean True(w/ seasonality) or False(w/o seasonality) and array of seasonal periods present in data
        """
        df = TimeSeries.from_dataframe(data_df1_series_dt,
                                       time_col=time_col,
                                       value_cols=value_col,
                                       fill_missing_dates=True,
                                       freq=None)

        periods = []
        for i in range(2, 400):
            is_seasonal, mseas = check_seasonality(df,
                                                   m=i,
                                                   max_lag=400)
            if is_seasonal:
                periods.append(mseas)

        if len(periods) > 0:
            return True, periods
        else:
            return False, periods


    seasonality, all_seasonality = seasonality_test(data_df1,
                                                    chosen_date1,
                                                    chosen_target1)

    seasonal, trend, residual = decompose(data_df1_series,
                                          chosen_target1)

    stationarity_data1 = test_stationarity(data_df1_series[chosen_target1])

    # Ljung-Box test for white noise
    def white_noise_test(
            df: pd.DataFrame,
            target: str):
        """
        Performs Ljung-Box test to check noise in time series data

        :param df: pandas dataframe
        :param target: target to check for white noise
        :return: boolean True(White noise) or False(No White noise)
        """
        p_val = acorr_ljungbox(df[target]).iloc[0]['lb_pvalue']
        if p_val > 0.05:
            return True
        else:
            return False

    white_noise = white_noise_test(data_df1_series,
                                   chosen_target1)

    # Measures of central tendency
    data_stats = data_df1_series[chosen_target1].describe()
    mean_data1 = round(data_stats['mean'], 2)
    median_data1 = round(data_stats['50%'], 2)
    std_data1 = round(data_stats['std'], 2)

    with plot_tab:
        st.subheader(f"Plots for {data1.name}")

        def make_plot(
                name: str,
                x_data: pd.Series or np.array,
                y_data: pd.Series or np.array,
                x_title: str,
                y_title: str):
            """
            Create plotly graph object plots from given parameters

            :param name: name data to be shown
            :param x_data: pandas series or numpy array for horizontal (x-axis)
            :param y_data: pandas series or numpy array for vertical (y-axis)
            :param x_title: x-axis title
            :param y_title: y-axis title
            :return: plotly graph object plot
            """
            fig = go.Figure()
            fig.add_trace(go.Line(name=name,
                                  x=x_data,
                                  y=y_data))
            fig.update_xaxes(gridcolor='grey')
            fig.update_yaxes(gridcolor='grey')
            fig.update_layout(colorway=["#7EE3C9"],
                              font_color="white",
                              paper_bgcolor="#2E3136",
                              plot_bgcolor="#2E3136",
                              xaxis_title=x_title,
                              yaxis_title=y_title,
                              title=f"{name} Plot")

            st.plotly_chart(fig,
                            use_container_width=True)


        # Data 1 plot
        make_plot(data1.name,
                  data_df1_series.index,
                  data_df1_series[chosen_target1],
                  chosen_date1,
                  chosen_target1)

        if seasonality:
            # Seasonal plot
            make_plot("Seasonal",
                      seasonal.index,
                      seasonal,
                      seasonal.index.name,
                      seasonal.name)

        # Trend plot
        make_plot("Trend",
                  trend.index,
                  trend,
                  trend.index.name,
                  trend.name)

        # Residual plot
        make_plot("Residual",
                  residual.index,
                  residual,
                  residual.index.name,
                  residual.name)

    with stat_tab:
        st.header("Descriptive Statistics")

        # Show descriptive statistics for file 1
        st.metric("No. of Variables",
                  data_df1_shape[1])
        st.metric("No. of Observations",
                  data_df1_shape[0])
        st.metric("Mean",
                  mean_data1)
        st.metric("Median",
                  median_data1)
        st.metric("Standard Deviation",
                  std_data1)
        st.metric("Seasonality",
                  seasonality)
        help_stationary = "This tells whether the dataset has seasonality or trend. " \
                          "A dataset with trend or seasonality is not stationary"
        st.metric("Stationarity",
                  stationarity_data1,
                  help=help_stationary)
        help_white_noise = "The past values of the predictors cannot be used " \
                           "to predict the future values if white noise is present." \
                           " In other words, the time series uploaded is a random walk."
        st.metric("White Noise",
                  white_noise,
                  help=help_white_noise)

    with correlation_tab:
        st.subheader("Cross Correlation Plots")
        st.caption("Interpretation: Positive correlation value means that the variables move in the same direction. "
                 "On the other hand, negative correlation value means that the variables move in opposite directions.")

        modes = ["Auto", "Manual"]
        help_correlation = "Auto setting finds the lag that gives the highest positive/negative correlation automatically. " \
                           "Manual mode allows the user to chose the lags manually."
        correlation_mode = st.sidebar.selectbox("Choose method for finding the best correlation",
                                                modes,
                                                help=help_correlation)

        def find_best_lag(
                df: pd.DataFrame,
                var1: str,
                var2: str,
                alpha=0.05):
            """
            Returns the best lag for positive and negative correlation
            :param df: pandas dataframe
            :param var1: variable name for the first variable
            :param var2: variable name for the second variable
            :param alpha: value for significance threshold (default=0.05)
            :return:
            """
            x = df[var1] - df[var1].shift(-1).fillna(df[var1].mean())
            y = df[var2] - df[var2].shift(-1).fillna(df[var2].mean())
            res = []
            for i in range(df[var1].shape[0]):
                corr_res = [pearsonr(x, y.shift(-i).fillna(y.mean())), i]
                if corr_res[0][1] < alpha:
                    res.append(corr_res)

            res.sort()
            best_positive_corr_lag = res[-2][1]
            best_negative_corr_lag = res[0][1]

            return best_positive_corr_lag, best_negative_corr_lag


        def differenced_correlation(
                df: pd.DataFrame,
                target: str,
                feature: str,
                period: int
        ):
            if stationarity_data1:
                # If data is stationary, compute the correlation coefficient directly
                corr_user = pearsonr(df[target].fillna(0),
                                     df[feature].shift(periods=-1 * period).fillna(0))
            else:
                # Stationarize time series then calculate correlation
                differenced_target = df[target] - df[target].shift(-1).fillna(df[target].mean())
                differenced_feature = df[feature] - df[feature].shift(-1).fillna(df[feature].mean())
                corr_user = pearsonr(differenced_target,
                                     differenced_feature.shift(periods=-1 * period).fillna(differenced_feature.mean()))

            return corr_user

        def strength(x):
            if abs(x) <= 0.3:
                return "Weak"
            elif 0.3 < abs(x) <= 0.7:
                return "Moderate"
            elif abs(x) > 0.7:
                return "Strong"

        def useful(x):
            if x < 0.05:
                return "significant"
            else:
                return "insignificant"

        # Cross correlation plots
        def make_correlation_plot(
                df: pd.DataFrame,
                target: str,
                feature: str,
                period: int,
                date: str,
                data_name: str,
                name: str):
            """
            Creates cross correlation and autocorrelation plots for time series data with corresponding lags

            :param df: input data in dataframe
            :param target: target name
            :param feature: feature name
            :param period: lag/shift to use
            :param date: chosen date column
            :param data_name: data name to be displayed in plot
            :param name: for title of plot
            :return: cross correlation and autocorrelation plot depending on the feature name
            """
            corr_user = differenced_correlation(df, target, feature, period)

            fig = go.Figure()
            fig.add_trace(go.Line(name=target,
                                  x=df.index,
                                  y=df[target]))
            fig.add_trace(go.Line(name=data_name,
                                  x=df.index,
                                  y=df[feature].shift(periods=-1 * period)))

            fig.update_xaxes(gridcolor="grey")
            fig.update_yaxes(gridcolor="grey")
            fig.update_layout(xaxis_title=date,
                              yaxis_title="Data",
                              font_color="white",
                              paper_bgcolor="#2E3136",
                              plot_bgcolor="#2E3136",
                              colorway=["#7EE3C9", "#70B0E0"],
                              title=f"{name}: {round(corr_user[0], 2)} ({strength(corr_user[0])} "
                                    f"and {useful(corr_user[1])})")

            st.plotly_chart(fig,
                            use_container_width=True)

        for feat in data_df1_series.columns:
            lag_user = st.number_input(f"Cross correlation lag/shift for {feat}",
                                       step=1,
                                       key=feat)
            if lag_user > 0:
                data_name = f"Shifted {feat}"
            else:
                data_name = feat

            if correlation_mode == 'Auto':
                pos_lag = find_best_lag(data_df1_series, chosen_target1, feat)[0]
                neg_lag = find_best_lag(data_df1_series, chosen_target1, feat)[1]

                st.markdown(f"<b><i>Use lag = {pos_lag} for best positive correlation "
                            f"and lag = {neg_lag} for best negative correlation</b></i>",
                            unsafe_allow_html=True)

            if feat == chosen_target1:
                name = "Autocorrelation"
                make_correlation_plot(data_df1_series,
                                      chosen_target1,
                                      feat,
                                      lag_user,
                                      chosen_date1,
                                      data_name,
                                      name)

            else:
                name = "Data Correlation"
                make_correlation_plot(data_df1_series,
                                      chosen_target1,
                                      feat,
                                      lag_user,
                                      chosen_date1,
                                      data_name,
                                      name)

    with forecast_tab:
        # Create autoML model for forecasting
        if st.button("Forecast"):
            def modeling():
                model = AutoTS(
                    forecast_length=10,
                    frequency='infer',
                    prediction_interval=0.95,
                    ensemble=None,
                    model_list='fast',
                    max_generations=10,
                    num_validations=1,
                    no_negatives=True
                )
                model = model.fit(data_df1_series)
                return model


            model = modeling()
            model_name1 = model.best_model_name
            prediction = model.predict()

            x_data1 = prediction.forecast.index
            y_data1 = prediction.forecast[chosen_target1].values
            y_upper1 = prediction.upper_forecast[chosen_target1].values
            y_lower1 = prediction.lower_forecast[chosen_target1].values

            # Forecast 1 plot
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                name="Data",
                x=data_df1_series.index,
                y=data_df1_series[chosen_target1]
            ))

            fig5.add_trace(go.Scatter(
                name='Prediction',
                x=x_data1,
                y=y_data1,
                # mode='lines',
                line=dict(color='rgb(31, 119, 180)'),
            ))

            fig5.add_trace(go.Scatter(
                name='Upper Bound',
                x=x_data1,
                y=y_upper1,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ))

            fig5.add_trace(go.Scatter(
                name='Lower Bound',
                x=x_data1,
                y=y_lower1,
                marker=dict(color="#70B0E0"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(150, 150, 150, 0.3)',
                fill='tonexty',
                showlegend=False
            ))

            fig5.update_xaxes(gridcolor='grey')
            fig5.update_yaxes(gridcolor='grey')
            fig5.update_layout(xaxis_title=chosen_date1,
                               yaxis_title=chosen_target1,
                               font_color="white",
                               paper_bgcolor="#2E3136",
                               plot_bgcolor="#2E3136",
                               title=f"{data1.name} Forecast using {model_name1}",
                               hovermode="x",
                               colorway=["#7EE3C9"])

            st.plotly_chart(fig5,
                            use_container_width=True)

except (NameError, IndexError, KeyError) as e:
    pass

print("Done Rendering Application!")

st.write("---")
end = time.time()
execution_time = end - start
st.write(f"Execution Time: {round(execution_time, 2)} seconds")
