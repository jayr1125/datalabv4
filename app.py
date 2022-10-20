import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import ruptures as rpt
from PIL import Image
from autots import AutoTS
from autots.tools.shaping import infer_frequency
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch.unitroot.cointegration import phillips_ouliaris
from darts.utils.statistics import check_seasonality
from darts import TimeSeries
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# Start of execution time calculation
start = time.time()

# Setting the app layout
st.set_page_config(layout="wide")

# Display FLNT logo
image = Image.open(r"flnt.png")
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

    help_dependent = "Dependent variable is the effect. It is the value that you are trying to forecast"
    help_independent = "Independent variable is the cause. It is the value which may contribute to the forecast"
    chosen_target1 = st.sidebar.selectbox("Choose dependent variable",
                                          targets1,
                                          help=help_dependent)
    features1 = list(data_df1_types.keys())
    features1.remove(chosen_target1)
    chosen_date1 = st.sidebar.selectbox("Choose date column to use",
                                        features1)
    chosen_features1 = st.sidebar.multiselect("Choose independent variable(s) to use",
                                              features1,
                                              help=help_independent)

    st.sidebar.write("---")

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
    
    # For darts time series dataframe
    data_df1_series_dt = data_df1_series.copy()

    data_df1_series.set_index(data_df1_series[chosen_date1],
                              inplace=True)
    data_df1_series.drop(chosen_date1,
                         axis=1,
                         inplace=True)

    # Get the inferred frequency of the dataset uploaded
    inferred_frequency = infer_frequency(data_df1_series)
    st.sidebar.write(f"Inferred Frequency of Dataset Uploaded: {inferred_frequency}")

    st.sidebar.write("---")

    # Maximum number of lags for Granger-causality Test
    granger_lag = st.sidebar.number_input("Max lag for Granger-causality test",
                                          step=1,
                                          value=5,
                                          help="Maximum number of lag to use for checking causality of two time series")

    # Create tabs for plots and statistics
    plot_tab, stat_tab, forecast_tab, prescriptive_tab = st.tabs(["Plots",
                                                                  "Statistics",
                                                                  "Forecast",
                                                                  "Prescriptive"])

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
    
    # Test for seasonality
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

    seasonal, trend, residual = decompose(data_df1_series, chosen_target1)

    stationarity_data1 = test_stationarity(data_df1_series[chosen_target1])

    # Measures of central tendency
    data_stats = data_df1_series[chosen_target1].describe()
    mean_data1 = round(data_stats['mean'], 2)
    median_data1 = round(data_stats['50%'], 2)
    std_data1 = round(data_stats['std'], 2)

    with stat_tab:
        st.header("Descriptive Statistics")
        st.write("---")

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
        
        st.write("---")

        st.subheader("Cross Correlation Plots")
        st.write("NOTE: The correlation values shown are based on the stationary version of the time series data."
                 " If the p-value is greater than 0.05, then the correlation value is statistically insignificant.")

        # Cross correlation plots
        for feat in data_df1_series.columns:
            lag_user = st.number_input(f"Cross correlation lag/shift for {feat}",
                                       step=1,
                                       key=feat)

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
                fig = go.Figure()
                fig.add_trace(go.Line(name=target,
                                      x=df.index,
                                      y=df[target]))
                fig.add_trace(go.Line(name=data_name,
                                      x=df.index,
                                      y=df[feature].shift(periods=-1*period)))

                if stationarity_data1:
                    # If data is stationary, compute the correlation coefficient directly
                    corr_user = pearsonr(df[target].fillna(0),
                                         df[feature].shift(periods=-1*period).fillna(0))
                else:
                    # Stationarize time series then calculate correlation
                    #differenced_target = df[target] - 2*df[target].shift(1) + df[target].shift(2)
                    differenced_target = df[target] - df[target].shift(1)
                    #differenced_feature = df[feature] - 2*df[feature].shift(1) + df[feature].shift(2)
                    differenced_feature = df[feature] - df[feature].shift(1)
                    corr_user = pearsonr(differenced_target.fillna(differenced_target.mean()),
                                         differenced_feature.shift(periods=-1 * period).fillna(differenced_feature.mean()))
                    
                fig.update_xaxes(gridcolor="grey")
                fig.update_yaxes(gridcolor="grey")
                fig.update_layout(xaxis_title=date,
                                  yaxis_title="Data",
                                  font_color="white",
                                  paper_bgcolor="#2E3136",
                                  plot_bgcolor="#2E3136",
                                  colorway=["#7EE3C9", "#70B0E0"],
                                  title=f"{name}: {round(corr_user[0], 2)} | p-value: {round(corr_user[1], 3)}")

                st.plotly_chart(fig,
                                use_container_width=True)

            if lag_user > 0:
                data_name = f"Shifted {feat}"
            else:
                data_name = feat

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
                
        st.write("---")

        st.subheader("Change Point Plot")
        
        algorithm_option = st.sidebar.selectbox("Choose algorithm to use for change point detection",
                                                ("Pelt", "Binary Segmentation", "Window"))

        # Change point plot
        def change_point_plot(
                data: pd.Series or np.array,
                target: str,
                algorithm: str):
            """
            Creates a plot of the input data with predicted change points based on the chosen algorithm
            :param data: a pandas series or numpy array
            :param target: target column
            :param algorithm: algorithm to detect change points,
            default="Pelt", options: "Pelt, "Binseg", "Window", "Dynp"
            :return: change point plot with break points
            """
            models_dict = {"Pelt": [rpt.Pelt, "rbf"],
                           "Binary Segmentation": [rpt.Binseg, "l2"],
                           "Window": [rpt.Window, "l2"]}

            model = models_dict[algorithm][1]

            if algorithm == "Pelt":
                algo = models_dict[algorithm][0](model=model).fit(data)
                my_bkps = algo.predict(pen=10)
            elif algorithm == "Window":
                algo = models_dict[algorithm][0](width=40, model=model).fit(data)
                my_bkps = algo.predict(n_bkps=10)
            else:
                algo = models_dict[algorithm][0](model=model).fit(data)
                my_bkps = algo.predict(n_bkps=10)

            fig = go.Figure()
            fig.add_trace(go.Line(name="Data",
                                  x=data.index,
                                  y=data[target]))
            for i in my_bkps[0:-1]:
                fig.add_vline(x=data.iloc[i].name, line_width=1, line_dash="dash", line_color="grey")
                
            fig.update_xaxes(gridcolor="#2E3136")
            fig.update_yaxes(gridcolor="grey")
            fig.update_layout(colorway=["#7EE3C9"],
                              font_color="white",
                              paper_bgcolor="#2E3136",
                              plot_bgcolor="#2E3136",
                              xaxis_title=data.index.name,
                              yaxis_title=chosen_target1,
                              title=f"Change Point Plot for {data1.name}")

            #st.plotly_chart(fig,
            #                use_container_width=True)
            return fig

        st.plotly_chart(change_point_plot(data_df1_series, 
                                          chosen_target1, 
                                          algorithm_option), 
                        use_container_width=True)
        
        st.write("---")
        
        st.subheader("Feature Importance Plot")
          
        def feature_importance_plot():
            dat = pd.DataFrame()
            for i in range(25, 0, -1):
                dat['t-' + str(i)] = data_df1_series.shift(i).values[:, 0]

            dat['t-'] = data_df1_series.values[:, 0]
            dat = dat[25:]
            array = dat.values
            x = array[:, 0:-1]
            y = array[:, -1]

            model = RandomForestRegressor(n_estimators=500, random_state=1)
            model.fit(x, y)
            names = dat.columns.values[0:-1]

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Feature importance",
                                 x=names[::-1],
                                 y=model.feature_importances_[::-1]))

            fig.update_xaxes(gridcolor="#2E3136")
            fig.update_yaxes(gridcolor="grey")
            fig.update_layout(xaxis_title=f"Lagged {chosen_target1}",
                              yaxis_title="Feature Importance",
                              font_color="white",
                              paper_bgcolor="#2E3136",
                              plot_bgcolor="#2E3136",
                              title=f"Feature Importance of Lagged {chosen_target1}",
                              colorway=["#7EE3C9"])

            #st.plotly_chart(fig,
            #                use_container_width=True)
            return fig

        st.metric("Feature Importance",
                  "",
                  help="This measures the importance of the lagged dependent variable for forecasting. Highest value is 1.")

        st.plotly_chart(feature_importance_plot(),
                        use_container_width=True)
        
        st.write("---")
        
        st.subheader("Outlier Detection")
        
        window = st.sidebar.number_input("Window Size for Outlier Detection",
                                         step=1,
                                         value=5,
                                         help="A smaller value makes the bands tighter, a larger value makes the bands smoother")
        
        # Outlier detection plot
        def outlier_plot(window_size):
            """
            Creates a plot for outlier detection based on window size and confidence intervals
            :param window_size: window size for outlier detection, smaller value makes the bands tighter, larger value
            makes the bands smoother. Default is 5
            :return: outlier plot
            """
            window_percentage = window_size
            k = int(len(data_df1_series_dt[chosen_target1]) * (window_percentage / 2 / 100))
            N = len(data_df1_series_dt[chosen_target1])
            
            get_bands = lambda data: (np.mean(data) + 3 * np.std(data), np.mean(data) - 3 * np.std(data))
            bands = [get_bands(data_df1_series_dt[chosen_target1][range(0 if i - k < 0 else i - k, i + k if i + k < N else N)]) for i in range(0, N)]
            upper, lower = zip(*bands)
            
            mask = (data_df1_series_dt[chosen_target1] > upper) | (data_df1_series_dt[chosen_target1] < lower)
            outlier = data_df1_series_dt[mask]

            fig = go.Figure()
            fig.add_trace(go.Line(name="Data",
                                  x=data_df1_series_dt[chosen_date1],
                                  y=data_df1_series_dt[chosen_target1]))
            fig.add_trace(go.Scatter(name="Outlier",
                                     x=outlier[chosen_date1],
                                     y=outlier[chosen_target1],
                                     mode="markers",
                                     marker=dict(color='red')))
            fig.add_trace(go.Scatter(name="Upper",
                                     x=data_df1_series_dt[chosen_date1],
                                     y=upper,
                                     mode='lines',
                                     showlegend=False,
                                     line=dict(color="#70B0E0")))
            fig.add_trace(go.Scatter(name="Lower",
                                     x=data_df1_series_dt[chosen_date1],
                                     y=lower,
                                     mode='lines',
                                     fillcolor='rgba(150, 150, 150, 0.3)',
                                     fill='tonexty',
                                     showlegend=False,
                                     line=dict(color="#70B0E0")))

            fig.update_xaxes(gridcolor="grey")
            fig.update_yaxes(gridcolor="grey")
            fig.update_layout(xaxis_title=chosen_date1,
                              yaxis_title=chosen_target1,
                              font_color="white",
                              paper_bgcolor="#2E3136",
                              plot_bgcolor="#2E3136",
                              title=f"Outlier Plot of {chosen_target1}",
                              colorway=["#7EE3C9"])

            return fig

        st.plotly_chart(outlier_plot(window),
                        use_container_width=True)

    with forecast_tab:
        # Create autoML model for forecasting
        data1_slider = st.sidebar.number_input("Forecast Horizon",
                                               min_value=1,
                                               value=10,
                                               step=1)
        if st.button("Forecast"):
            def modeling(slider):
                model = AutoTS(
                    forecast_length=slider,
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

            model = modeling(data1_slider)
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

    with prescriptive_tab:
        if seasonality:
            st.metric("Seasonalities Present: ",
                      str(all_seasonality[:11]),
                      help="Using these values as lags for cross/auto correlation would yield a high correlation value")

        # Granger-causality test
        differenced_gc = data_df1_series[chosen_target1] - 2*data_df1_series[chosen_target1].shift(1) + data_df1_series[chosen_target1].shift(2)
        differenced_gc = differenced_gc.fillna(differenced_gc.mean())
        gc = dict()
        for feature in data_df1_series.columns:
            gc_lag = []
            for i in range(1, granger_lag + 1):
                p_val = grangercausalitytests(pd.DataFrame(zip(data_df1_series[feature], differenced_gc)),
                                              maxlag=granger_lag,
                                              verbose=False)[i][0]['ssr_ftest'][1]
                if (p_val < 0.05) and (feature != chosen_target1):
                    gc_lag.append(i)
                    gc[feature] = gc_lag

        for feat_gc, lag_gc in gc.items():
            st.metric(f"{feat_gc} is useful in forecasting the future values of {chosen_target1} at these lags",
                      str(lag_gc))

        # Phillips-Ouliaris Test for cointegration
        coint_feat = list()
        if not stationarity_data1:
            for feat in data_df1_series.columns:
                if feat != chosen_target1:
                    po_test = phillips_ouliaris(data_df1_series[chosen_target1],
                                                data_df1_series[feat])
                    if po_test.pvalue > 0.05:
                        coint_feat.append(feat)

        if len(coint_feat) > 0:
            st.metric(f"{chosen_target1} is cointegrated with the following variable(s)",
                      str(coint_feat),
                      help="Cointegrated means that the two variables have a relationship/correlation in the long term.")

except (NameError, IndexError, KeyError) as e:
    pass

print("Done Rendering Application!")

st.write("---")
end = time.time()
execution_time = end - start
st.write(f"Execution Time: {round(execution_time, 2)} seconds")
