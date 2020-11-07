from sklearn.base import BaseEstimator, TransformerMixin


class GenreDummies(BaseEstimator, TransformerMixin):
    def fit(self, X, y): return self

    def transform(self, X):
        res = []
        for e in X:
            res.append({g: 1 for g in e['genres']})
        return res