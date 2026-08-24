from __future__ import annotations

import numpy as np
from Neural_Decoding.decoders import KalmanFilterDecoder

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # Code/ for local imports
from decoder_utils import r2_score_1d


def shape(name, value):
    print(f"{name:<28} {np.shape(value)}")


def make_toy_data(n_time=500, n_states=2, n_units=12, random_seed=0):
    # Make a smooth hidden movement state.
    rng = np.random.default_rng(random_seed)
    state = np.zeros((n_time, n_states), dtype=float)
    state[0] = rng.normal(size=n_states)
    A = np.array([[0.95, 0.04], [-0.03, 0.92]])
    for t in range(1, n_time):
        state[t] = A @ state[t - 1] + rng.normal(scale=0.15, size=n_states)

    # Make neural activity as a noisy linear readout of the state.
    H = rng.normal(scale=0.8, size=(n_states, n_units))
    neural = state @ H + rng.normal(scale=0.3, size=(n_time, n_units))
    return neural, state


def print_training_shapes(model, X_train, y_train):
    # Kording package stores model.model = [A, W, H, Q].
    # In the package notation, y is movement state and X is neural activity.
    A, W, H, Q = model.model
    X = np.asmatrix(y_train.T)
    Z = np.asmatrix(X_train.T)
    X1 = X[:, :-1]
    X2 = X[:, 1:]

    print("\nTraining matrices")
    print("-" * 78)
    shape("neural X_train", X_train)
    shape("state y_train", y_train)
    shape("Z = X_train.T", Z)
    shape("X = y_train.T", X)
    shape("X1 = X[:, :-1]", X1)
    shape("X2 = X[:, 1:]", X2)
    shape("A", A)
    shape("W", W)
    shape("H", H)
    shape("Q", Q)
    print("A formula: X2 @ X1.T @ inv(X1 @ X1.T)")
    shape("X2 @ X1.T", X2 @ X1.T)
    shape("X1 @ X1.T", X1 @ X1.T)
    print("H formula: Z @ X.T @ inv(X @ X.T)")
    shape("Z @ X.T", Z @ X.T)
    shape("X @ X.T", X @ X.T)


def print_prediction_shapes(model, X_test, y_test):
    A, W, H, Q = model.model
    Z = np.asmatrix(X_test.T)
    X = np.asmatrix(y_test.T)
    num_states = X.shape[0]
    P = np.asmatrix(np.zeros((num_states, num_states)))
    state = X[:, 0]

    P_m = A * P * A.T + W
    state_m = A * state
    innovation_cov = H * P_m * H.T + Q
    K = P_m * H.T * np.linalg.inv(innovation_cov)
    innovation = Z[:, 1] - H * state_m
    state_next = state_m + K * innovation

    print("\nPrediction matrices for one update step")
    print("-" * 78)
    shape("Z = X_test.T", Z)
    shape("X = y_test.T", X)
    shape("initial state", state)
    shape("P", P)
    shape("P_m = A @ P @ A.T + W", P_m)
    shape("state_m = A @ state", state_m)
    shape("innovation_cov", innovation_cov)
    shape("K", K)
    shape("innovation", innovation)
    shape("state_next", state_next)


def main():
    # Kording package convention:
    # X is neural data, y is the output/state being decoded.
    neural, state = make_toy_data()
    split = int(len(neural) * 0.7)
    X_train = neural[:split]
    y_train = state[:split]
    X_test = neural[split:]
    y_test = state[split:]

    model = KalmanFilterDecoder(C=1)
    model.fit(X_train, y_train)
    print_training_shapes(model, X_train, y_train)
    print_prediction_shapes(model, X_test, y_test)
    y_pred = model.predict(X_test, y_test)

    r2s = np.asarray([
        r2_score_1d(y_test[:, dim], y_pred[:, dim])
        for dim in range(y_test.shape[1])
    ])
    print("Toy Kording Kalman")
    print("=" * 78)
    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)
    print("R2s:", r2s)
    print(f"mean_R2: {np.nanmean(r2s):.6f}")


if __name__ == "__main__":
    main()
