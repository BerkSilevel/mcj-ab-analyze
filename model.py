import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class DecayRetentionModel(BaseEstimator, RegressorMixin):
    """
    D1 ve D3 retention oranlarından D4–D15 retention tahmini yapan decay tabanlı model.
    """

    def __init__(self):
        pass

    def fit(self, X=None, y=None):
        # Model sabit decay kuralına göre çalıştığı için fit gerekli değil
        return self

    def predict(self, X):
        """
        Parametreler:
            X: np.ndarray veya DataFrame gibi (n_samples, 2) boyutlu D1 ve D3 retention oranları

        Dönüş:
            np.ndarray (n_samples, 12) boyutlu D4–D15 tahminleri
        """
        X = np.array(X)
        D1 = X[:, 0]
        D3 = X[:, 1]

        predictions = []

        for d1, d3 in zip(D1, D3):
            # Hatalı değer kontrolü
            if d1 <= 0 or d3 <= 0 or d3 > d1:
                predictions.append([0.0] * 12)
                continue

            # Decay oranı (günlük) — D1'den D3'e geçişteki azalma
            decay_rate = np.log(d3 / d1) / 2  # çünkü 2 gün var
            daily_decay = np.exp(decay_rate)

            # D4–D15 tahminleri
            preds = []
            for day in range(4, 16):
                r = d3 * (daily_decay ** (day - 3))
                preds.append(r)

            predictions.append(preds)

        return np.array(predictions)
