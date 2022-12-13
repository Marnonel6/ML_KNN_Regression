import numpy as np
from src import generate_regression_data, PolynomialRegression, KNearestNeighbor


def test_polynomial_regression_basics():
    degrees = range(10)
    x, y = generate_regression_data(3, 10, amount_of_noise=1)
    for degree in degrees:
        model = PolynomialRegression(degree)
        model.fit(x, y)
        msg = f"With degree {degree}, model.w should have shape ({degree + 1}, 1)"
        assert model.w.shape == (degree + 1, 1), msg

        # Test model.predict
        coefs = np.random.uniform(-2, 2, size=(degree + 1, 1))
        model.w = coefs
        preds = model.predict(x).reshape(-1)
        # Do not use any of numpy's polynomial functions in your code.
        comparison = np.polynomial.Polynomial(coefs.reshape(-1))
        ref = comparison(x).reshape(-1)
        assert np.all(np.isclose(preds, ref)), "Predict should match reference"


def test_polynomial_regression_full():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    degrees = range(10)
    amounts = [10, 100, 1000, 10000]

    for degree in degrees:
        p = PolynomialRegression(degree)
        for amount in amounts:
            x, y = generate_regression_data(degree, amount, amount_of_noise=0.0)
            p.fit(x, y)
            y_hat = p.predict(x)
            mse = mean_squared_error(y, y_hat)
            # print(f"y = {y}")
            # print(f"y_hat = {y_hat}")
            msg = f"With degree {degree} and amount {amount}, mse {mse:.3f} > 0.1"
            assert mse < 1e-1, msg

            for other_degree in degrees:
                p2 = PolynomialRegression(other_degree)
                p2.fit(x, y)
                p2_preds = p2.predict(x)
                p2_mse = mean_squared_error(p2_preds, y)
                
                poly_features = PolynomialFeatures(other_degree)
                linreg = LinearRegression()
                linreg.fit(poly_features.fit_transform(x), y)
                linreg_preds = linreg.predict(poly_features.transform(x))
                linreg_mse = mean_squared_error(linreg_preds, y)

                msg = f"With degree {degree} and amount {amount}, {p2_mse:.3f} != {linreg_mse:.3f}"
                assert np.abs(linreg_mse - p2_mse) < 0.1, msg
