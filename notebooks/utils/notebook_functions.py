import numpy as np
import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
import pandas as pd
import plotly.subplots as sp
from interpret import show
from math import ceil
from interpret.blackbox import PartialDependence
from PIL import Image

def generate_predictions(model, df, target_column='target'):
    X_train = df.drop(target_column, axis=1)
    predictions = model.predict(X_train)
    df['Prediction'] = predictions
    return df

def get_variable_importances(ebm_model):
    # Obter a explicação global do modelo
    global_explanation = ebm_model.explain_global()

    # Extrair a importância das variáveis e seus nomes
    importances = global_explanation.data()['scores']
    variable_names = global_explanation.data()['names']

    # Criar um DataFrame para facilitar a ordenação
    df_importances = pd.DataFrame({'variable': variable_names, 'importance': importances})

    # Ordenar pela magnitude da importância (valores absolutos)
    df_importances = df_importances.reindex(df_importances.importance.abs().sort_values(ascending=False).index)
    
    return df_importances

def plot_sorted_predictions_grid(dfs, models, titles, explain_models, target_column='target', prediction_column='Prediction', grid_title='Model Comparison'):
    # Definindo o layout da figura com 3 linhas e 2 colunas
    fig = sp.make_subplots(
        rows=3, cols=2, subplot_titles=titles,
        horizontal_spacing=0.1, vertical_spacing=0.1
    )

    # Adicionando gráficos de dispersão e feature importance
    for i, (df, model) in enumerate(zip(dfs, models), start=1):
        if prediction_column not in df.columns:
            df = generate_predictions(model, df, target_column)

        df_sorted = df.dropna(subset=[target_column, prediction_column])
        df_sorted = df_sorted.sort_values(by=target_column).reset_index(drop=True)

        rmse = np.sqrt(mean_squared_error(df_sorted[target_column], df_sorted[prediction_column]))
        mae = mean_absolute_error(df_sorted[target_column], df_sorted[prediction_column])
        r2 = r2_score(df_sorted[target_column], df_sorted[prediction_column])

        # Configurando a visibilidade da legenda
        showlegend = True if i == 1 else False

        trace1 = go.Scatter(x=df_sorted.index, y=df_sorted[target_column], mode='lines', name='Actual', line=dict(color='blue'), showlegend=showlegend)
        trace2 = go.Scatter(x=df_sorted.index, y=df_sorted[prediction_column], mode='lines', name='Predicted', line=dict(color='red'), showlegend=showlegend)

        row, col = (i-1) // 2 + 1, (i-1) % 2 + 1
        fig.add_trace(trace1, row=row, col=col)
        fig.add_trace(trace2, row=row, col=col)

        fig.add_annotation(
            text=f'RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}',
            xref="x domain", yref="y domain",
            x=0.5, y=-0.25, showarrow=False,
            xanchor='center', yanchor='bottom',
            font=dict(size=12),
            row=row, col=col
        )
        if i <= 2:  # Ajuste conforme necessário
            row = 3
            col = i
            if explain_models is not None:
                feature_data = explain_models[i-1]  
                feature_names = feature_data['feature_names']
                feature_scores = feature_data['feature_scores']
            else:
                feature_names = []
                feature_scores = []

            ebm_df = pd.DataFrame({'Feature': feature_names, 'Score': feature_scores})
            ebm_df_sorted = ebm_df.sort_values(by='Score', ascending=False).head(15)  # Top 15 features

            feature_importance_trace = go.Bar(x=ebm_df_sorted['Feature'], y=ebm_df_sorted['Score'])
            fig.add_trace(feature_importance_trace, row=row, col=col)


    # Ajustes finais para a figura
    fig.update_layout(
        title_text=grid_title, title_x=0.5,
        height=1200, width=900, showlegend=True
    )

    # Mostrando a figura
    fig.show()


def plot_sorted_predictions_plotly(df, target_column='target', prediction_column='Prediction', title=''):
    # Check if DataFrame contains specified columns
    if target_column not in df.columns or prediction_column not in df.columns:
        raise ValueError("The DataFrame does not contain the specified columns.")

    # Drop NaN values
    df = df.dropna(subset=[target_column, prediction_column])

    # Sort and reset index
    df_sorted = df.sort_values(by=target_column).reset_index()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(df_sorted[target_column], df_sorted[prediction_column]))
    mae = np.mean(np.abs(df_sorted[target_column] - df_sorted[prediction_column]))
    r2 = r2_score(df_sorted[target_column], df_sorted[prediction_column])

    # Actual and predicted values traces
    trace1 = go.Scatter(x=df_sorted.index, y=df_sorted[target_column], mode='lines', name='Actual', line=dict(color='blue'))
    trace2 = go.Scatter(x=df_sorted.index, y=df_sorted[prediction_column], mode='lines', name='Predicted', line=dict(color='red'))

    # Linear regression for the predicted values
    m, b = np.polyfit(df_sorted.index, df_sorted[prediction_column], 1)
    regression_line = m * df_sorted.index + b

    # Trend line (linear regression) trace
    trace3 = go.Scatter(x=df_sorted.index, y=regression_line, mode='lines', name='Trend Line (Linear Regression)', line=dict(color='green', dash='dot'))

    # Layout including metrics
    layout = go.Layout(
        title=f'{title}<br>RMSE: {rmse:.4f}<br>MAE: {mae:.4f}<br>R2: {r2:.4f}',
        xaxis=dict(title='Index'),
        yaxis=dict(title='Value'),
        hovermode='closest',
        template='plotly_white'
    )

    # Create figure and show plot
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig.show()

def display_test_result(df: pd.DataFrame, datetime_index=True):

    if (datetime_index):
        index = df.index
    else:
        index = [*range(len(df.index))]

    feature1 = 'Prediction'
    target_variable = 'target'
    upper_ci = 'Min'
    lower_ci = 'Max'

    # Remove rows with NaN values in target_variable and feature2
    df_metric = df.dropna(subset=[target_variable, feature1, upper_ci, lower_ci])

    # Calculate RMSE, MAPE, and R2
    rmse = np.round(sqrt(mean_squared_error(
        df_metric[target_variable], df_metric[feature1])), 4)
    r2 = np.round(r2_score(df_metric[target_variable], df_metric[feature1]), 4)
    mape = np.round(
        np.mean(np.abs((df_metric[target_variable] - df_metric[feature1]))), 4)

    # Create a Plotly figure with three traces (one for each variable) on subplots
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=index, y=df[target_variable], mode='lines',
                             name=target_variable, yaxis='y',
                             line=dict(color='#247454')))
    
    # Add confidence interval for target_variable
    fig.add_trace(go.Scatter(
        x=index + index[::-1],
        y=pd.concat([df[upper_ci], df[lower_ci][::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval',
        showlegend=False
    ))

    # Third trace with color set to blue (for the third column)
    fig.add_trace(go.Scatter(x=index, y=df[feature1], mode='lines',
                             name=feature1, yaxis='y3',
                             line=dict(color='red')))

    # Find the range for all y-axes
    y_range = [min(df[target_variable].min(), df[feature1].min()),
               max(df[target_variable].max(), df[feature1].max())]

    # Customize the plot (optional)
    fig.update_layout(
        title='RMSE: ' + str(rmse) + '<br>' +
        'MAE: ' + str(mape) + '<br>' +
        'R2: ' + str(r2) + '<br>',
        xaxis=dict(title='DATA'),
        yaxis=dict(title=target_variable, showgrid=True, range=y_range),
        yaxis2=dict(title=feature1, showgrid=True,
                    overlaying='y', side='right', range=y_range),
        yaxis3=dict(title=feature1, showgrid=True,
                    overlaying='y', side='right', range=y_range),
        template='plotly_white'
    )

    # Display the plot
    fig.show()

def plot_sorted_predictions_plotly(df, target_column='target', prediction_column='Prediction', title=''):
    """
    This function takes in a DataFrame and the names of the actual and predicted value columns,
    sorts the DataFrame by the actual values, and plots these against the predicted values using Plotly.
    It also calculates RMSE, MAE, and R2 and adds them to the plot layout.
    Additionally, it plots a line between the first and last predicted values.
    
    :param df: DataFrame containing the columns with actual and predicted values.
    :param target_column: String name of the column with the actual values.
    :param prediction_column: String name of the column with the predicted values.
    :param title: Title for the plot.
    """
    # Contains the specified columns
    if target_column not in df.columns or prediction_column not in df.columns:
        raise ValueError("The DataFrame does not contain the specified columns.")

    # Drop NaN values from these columns to avoid plotting issues
    df = df.dropna(subset=[target_column, prediction_column])

    df_sorted = df.sort_values(by=target_column).reset_index()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(df_sorted[target_column], df_sorted[prediction_column]))
    mae = np.mean(np.abs(df_sorted[target_column] - df_sorted[prediction_column]))
    r2 = r2_score(df_sorted[target_column], df_sorted[prediction_column])

    trace1 = go.Scatter(
        x=df_sorted.index, 
        y=df_sorted[target_column], 
        mode='lines', 
        name='Actual', 
        line=dict(color='blue')
    )
    trace2 = go.Scatter(
        x=df_sorted.index, 
        y=df_sorted[prediction_column], 
        mode='lines', 
        name='Predicted', 
        line=dict(color='red')
    )

    # Plot a line between the first and last point of the predicted values
    trace3 = go.Scatter(
        x=[df_sorted.index[0], df_sorted.index[-1]],
        y=[df_sorted[prediction_column].iloc[0], df_sorted[prediction_column].iloc[-1]],
        mode='lines',
        name='Trend Line',
        line=dict(color='green', dash='dot')
    )

    # Layout for the plot, including metrics
    layout = go.Layout(
        title=f'{title}<br>RMSE: {rmse:.4f}<br>MAE: {mae:.4f}<br>R2: {r2:.4f}',
        xaxis=dict(title='Index'),
        yaxis=dict(title='Value'),
        hovermode='closest',
        template='plotly_white'
    )

    # Combine the traces and layout in a figure
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)

    # Show the plot
    fig.show()

def plot_comparison_prediction_intervals(df1, df2, target_column='target', prediction_column='Prediction', lower_bound_column='Min', upper_bound_column='Max', title1='', title2=''):
    def process_df(df, title, color):
        df = df.dropna(subset=[target_column, prediction_column, lower_bound_column, upper_bound_column])
        df_sorted = df.sort_values(by=target_column).reset_index()
        rmse = np.sqrt(mean_squared_error(df_sorted[target_column], df_sorted[prediction_column]))
        mae = np.mean(np.abs(df_sorted[target_column] - df_sorted[prediction_column]))
        r2 = r2_score(df_sorted[target_column], df_sorted[prediction_column])

        trace_actual = go.Scatter(x=df_sorted.index, y=df_sorted[target_column], mode='lines', name=f'{title} Actual', line=dict(color=color))
        trace_pred = go.Scatter(x=df_sorted.index, y=df_sorted[prediction_column], mode='markers', name=f'{title} Predicted', marker=dict(color=color, size=5))
        trace_lower = go.Scatter(x=df_sorted.index, y=df_sorted[lower_bound_column], mode='lines', name=f'{title} Lower Bound', line=dict(color=color, dash='dash'), showlegend=False)
        trace_upper = go.Scatter(x=df_sorted.index, y=df_sorted[upper_bound_column], mode='lines', name=f'{title} Upper Bound', line=dict(color=color, dash='dash'), fill='tonexty', showlegend=False)

        return [trace_actual, trace_pred, trace_lower, trace_upper], rmse, mae, r2

    traces1, rmse1, mae1, r21 = process_df(df1, title1, 'blue')
    traces2, rmse2, mae2, r22 = process_df(df2, title2, 'red')

    layout = go.Layout(
        title=f'Comparison<br>{title1} RMSE: {rmse1:.4f}, MAE: {mae1:.4f}, R2: {r21:.4f}<br>{title2} RMSE: {rmse2:.4f}, MAE: {mae2:.4f}, R2: {r22:.4f}',
        xaxis=dict(title='Index'),
        yaxis=dict(title='Value'),
        hovermode='closest',
        template='plotly_white'
    )

    fig = go.Figure(data=traces1 + traces2, layout=layout)
    fig.show()

def plot_prediction_interval(df, target_column='target', prediction_column='Prediction', lower_bound_column='Min', upper_bound_column='Max', title=''):
    required_columns = [target_column, prediction_column, lower_bound_column, upper_bound_column]
    if not all(column in df.columns for column in required_columns):
        raise ValueError("The DataFrame does not contain the required columns.")

    df = df.dropna(subset=required_columns)

    df_sorted = df.sort_values(by=target_column).reset_index()

    rmse = np.sqrt(mean_squared_error(df_sorted[target_column], df_sorted[prediction_column]))
    mae = np.mean(np.abs(df_sorted[target_column] - df_sorted[prediction_column]))
    r2 = r2_score(df_sorted[target_column], df_sorted[prediction_column])

    trace_pred = go.Scatter(x=df_sorted.index, y=df_sorted[prediction_column], mode='lines', name='Predicted', line=dict(color='red'))

    trace_interval = go.Scatter(
        x=df_sorted.index.tolist() + df_sorted.index.tolist()[::-1],  # x, then x reversed
        y=df_sorted[upper_bound_column].tolist() + df_sorted[lower_bound_column].tolist()[::-1],  # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Prediction Interval'
    )

    layout = go.Layout(
        title=f'{title}<br>RMSE: {rmse:.4f}<br>MAE: {mae:.4f}<br>R2: {r2:.4f}',
        xaxis=dict(title='Index'),
        yaxis=dict(title='Value'),
        hovermode='closest',
        template='plotly_white'
    )

    fig = go.Figure(data=[trace_pred, trace_interval], layout=layout)
    fig.show()