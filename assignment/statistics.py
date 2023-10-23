import itertools
from dataclasses import dataclass

import pandas
from pandas import DataFrame
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


def get_correlation(data: DataFrame) -> list[tuple[str, str, float]]:
    results = []
    columns = [col for col in data.columns]
    for first_index, column in enumerate(columns):
        for second_index in range(first_index + 1, len(columns)):
            results.append((
                columns[first_index],
                columns[second_index],
                data[columns[first_index]].corr(data[columns[second_index]])
            ))
    return results


def get_filtered_correlation(correlations: list[tuple[str, str, float]], threshold: float):
    relevant_correlations = list(filter(lambda c: c[2] >= threshold, correlations))
    print('Highly correlated training variables:')
    for relevant_correlation in relevant_correlations:
        print(f'{relevant_correlation[0]} & {relevant_correlation[1]}: {relevant_correlation[2]}')
    print('')
    return relevant_correlations


def get_p_values(data: DataFrame, target_column: str) -> dict[str, float]:
    result = {}
    columns = list(filter(lambda c: c != target_column, data.columns))
    for column in columns:
        _, p = stats.pearsonr(data[column], data[target_column])
        result[column] = p

    sorted_result = sorted(result.items(), key=lambda item: item[1], reverse=True)
    return {key: value for key, value in sorted_result}


def get_r2_values(data: DataFrame, target_column: str) -> dict[str, float]:
    result = {}
    columns = list(filter(lambda c: c != target_column, data.columns))
    for column in columns:
        model = create_lin_reg_model(data[column].to_numpy().reshape(-1, 1), data[target_column])
        prediction = model.predict(data[column].to_numpy().reshape(-1, 1))
        score = r2_score(y_true=data[target_column], y_pred=prediction)
        result[column] = score
    sorted_result = sorted(result.items(), key=lambda item: item[1], reverse=True)
    return {key: value for key, value in sorted_result}
    #columns = list(filter(lambda c: c != target_column, data.columns))
    #unsorted_p_values = {column: r2_score(data[target_column], data[column]) for column in columns}
    #sorted_p_values = sorted(unsorted_p_values.items(), key=lambda item: item[1], reverse=True)
    #return {key: value for key, value in sorted_p_values}


def split_data(data: DataFrame, target_column: str) -> (DataFrame, DataFrame):
    return data.drop(target_column, axis=1), data[target_column]


def create_lin_reg_model(train_data: DataFrame, train_y: DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(train_data, train_y)
    return model


@dataclass
class ModelPerformance:
    mse: float
    r2: float

    def __repr__(self):
        return f'(mse: {self.mse:.6f}, r2: {self.r2:.6f})'


def test_lin_reg_model(model: LinearRegression, test_data: DataFrame, test_y: DataFrame) -> ModelPerformance:
    prediction = model.predict(test_data)
    return ModelPerformance(
        mse=mean_squared_error(y_true=test_y, y_pred=prediction),
        r2=r2_score(y_true=test_y, y_pred=prediction)
    )


def evaluate_model_strategy(train: DataFrame, test: DataFrame, target_column: str, test_on_train: bool = False) -> list[tuple[list[str], ModelPerformance]]:
    results = []
    train_x, train_y = split_data(train, target_column)
    test_x, test_y = split_data(test, target_column)

    if test_on_train:
        test_x = train_x
        test_y = train_y

    columns = train_x.columns
    for r in range(1, len(columns) + 1):
        for combination in itertools.combinations(columns, r):
            selected_columns = list(combination)
            selected_train_x = train_x[selected_columns]
            selected_test_x = test_x[selected_columns]
            model = create_lin_reg_model(selected_train_x, train_y)
            performance = test_lin_reg_model(model, selected_test_x, test_y)
            results.append((selected_columns, performance))

    return results


def get_all_column_combinations(columns: list[str]):
    for r in range(1, len(columns) + 1):
        for combination in itertools.combinations(columns, r):
            yield combination


def sort_model_performances(performances: list[tuple[list[str], [ModelPerformance]]], by: str, limit: int = 0):
    reverse = True if by == 'r2' else False
    return list(sorted(performances, key=lambda p: getattr(p[1], by), reverse=reverse))[:limit]


def get_selected_features() -> list[str]:
    return ['Inj1PosVolAct_Var', 'Inj1PrsAct_meanOfInjPhase', 'Inj1HopTmpAct_1stPCscore',
            'ClpFceAct_1stPCscore', 'OilTmp1Act_1stPCscore']


def create_polynomial_feature(data: pandas.DataFrame,
                              degree: int,
                              bias: bool,
                              interaction_only: bool):
    features = get_selected_features()
    selected_features = data[features]

    polynomial = PolynomialFeatures(degree=degree, include_bias=bias, interaction_only=interaction_only)
    return polynomial.fit_transform(selected_features)


def create_higher_order_model(train_x: pandas.DataFrame,
                              train_y: pandas.DataFrame,
                              degree: int,
                              bias: bool,
                              interaction_only: bool):
    polynomial_features = create_polynomial_feature(train_x, degree, bias, interaction_only)
    model = LinearRegression()
    model.fit(polynomial_features, train_y)
    return model


@dataclass
class HigherOrderModelParameters:
    degree: int
    bias: bool
    interaction_only: bool

    def __repr__(self):
        return f'degree: {self.degree}, bias: {self.bias}, intercept only: {self.interaction_only}'


def evaluate_higher_order_model(train, test, target_column: str):
    train_x, train_y = split_data(train, target_column)
    test_x, test_y = split_data(test, target_column)

    parameter_combinations: list[HigherOrderModelParameters] = [
        HigherOrderModelParameters(degree=2, bias=False, interaction_only=True),
        HigherOrderModelParameters(degree=2, bias=True, interaction_only=True),
        HigherOrderModelParameters(degree=2, bias=False, interaction_only=False),
        HigherOrderModelParameters(degree=2, bias=True, interaction_only=False),
        HigherOrderModelParameters(degree=3, bias=False, interaction_only=True),
        HigherOrderModelParameters(degree=4, bias=False, interaction_only=True),
        HigherOrderModelParameters(degree=5, bias=False, interaction_only=True),
    ]

    for p in parameter_combinations:
        model = create_higher_order_model(train_x, train_y, p.degree, p.bias, p.interaction_only)
        polynomial_training = create_polynomial_feature(train_x, p.degree, p.bias, p.interaction_only)
        polynomial_testing = create_polynomial_feature(test_x, p.degree, p.bias, p.interaction_only)
        prediction_training = model.predict(polynomial_training)
        prediction_testing = model.predict(polynomial_testing)
        result_training = ModelPerformance(mse=mean_squared_error(train_y, prediction_training),
                                           r2=r2_score(train_y, prediction_training))
        result_testing = ModelPerformance(mse=mean_squared_error(test_y, prediction_testing),
                                          r2=r2_score(test_y, prediction_testing))
        result_difference = ModelPerformance(mse=result_testing.mse - result_training.mse,
                                             r2=result_training.r2 - result_testing.r2)

        print('MODEL PERFORMANCE:', p)
        print('Training:', result_training)
        print('Testing:', result_testing)
        print('Difference:', result_difference)
        print('')
