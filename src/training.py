from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(y_true, y_predicted):
    mae = metrics.mean_absolute_error(y_true, y_predicted)
    mse = metrics.mean_squared_error(y_true, y_predicted)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_predicted))
    r2_square = metrics.r2_score(y_true, y_predicted)

    return mae, mse, rmse, r2_square


def predict(model, x_train, y_train, x_test, args=None, kwargs=None):
    if not args: args = []
    if not kwargs: kwargs = dict()

    predict_model = model(*args, **kwargs)
    predict_model.fit(x_train, y_train)
    y_pred = predict_model.predict(x_test)
    return y_pred


def compare_models(x_train, y_train, x_test, y_test, models,
                   args=[], kwargs=dict(), show=True):
    df_scores = pd.DataFrame(columns=['model', 'mae', 'mse', 'rmse', 'r2_square'])
    for model in models:
        y_pred = predict(model, x_train, y_train, x_test, *args, **kwargs)
        results_append = pd.DataFrame(data=[[model.__name__, *evaluate(y_test, y_pred)]],
                                      columns=df_scores.columns)
        if show:
            # ax = sns.distplot(y_test, hist=False, color='r', label='Actual value')
            # sns.distplot(y_pred, hist=False, color='b', label='Predicted value', ax=ax)
            ax = sns.kdeplot(x=y_test, color='r', label='Actual value')
            sns.kdeplot(x=y_pred, color='b', label='Predicted value', ax=ax)
            ax.set(title=model.__name__)
            plt.show()

        df_scores = df_scores.append(results_append, ignore_index=True)
        print(f'{model.__name__} finished')

    return df_scores


def check_model_on_valid(model, x_train, y_train, x_valid, y_valid):

    y_pred = model().fit(x_train, y_train).predict(x_valid)
    y_true = y_valid

    ax = sns.kdeplot(x=y_true, color='r', label='Actual value')
    sns.kdeplot(x=y_pred, color='b', label='Predicted value', ax=ax)
    ax.set(title=model.__name__)
    plt.show()

    scores_ = pd.DataFrame(np.reshape(evaluate(y_true, y_pred), (1, -1)),
                           columns=['mae','mse','rmse','r2_square'])
    return scores_