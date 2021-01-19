from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CrewFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, field, min_cnt_movies=2):
        self.field = field
        self.min_cnt_movies = min_cnt_movies

    def fit(self, X, y):
        # Esto no es la forma mas elegante, pero es mas comodo y a esta altura priorizo la comodidad
        # Llevamos las cosas de nuevo a un DataFrame y calculamos features por director
        directors_stats = (
            pd.DataFrame(X)
                .groupby(self.field)
                .agg(
                    n_films=('tconst', 'count'),
                    min_rating=('averageRating', 'min'),
                    avg_rating=('averageRating', 'mean'),
                    max_rating=('averageRating', 'max'),
                    min_votes=('numVotes', 'min'),
                    avg_votes=('numVotes', 'mean'),
                    max_votes=('numVotes', 'max'),
            )
        )

        # Guardamos las estadisticas
        self.directors_stats_ = directors_stats

        # Diccionario con los datos para los directores comunes
        self.directors_stats_lk_ = (
            directors_stats[directors_stats.n_films >= self.min_cnt_movies].to_dict(orient='index')
        )

        # Valor default para los que consideramos que tenemos demasiado poca data
        self.default_ = directors_stats[directors_stats.n_films < self.min_cnt_movies].mean(0).to_dict()
        if self.min_cnt_movies > 1:
            self.default_ = directors_stats[directors_stats.n_films < self.min_cnt_movies].mean(0).to_dict()
        else:
            self.default_ = directors_stats.mean(0).to_dict()
        return self

    def transform(self, X):
        res = []
        for e in X:
            if e[self.field] in self.directors_stats_lk_:
                res.append(self.directors_stats_lk_[e[self.field]])
            else:
                res.append(self.default_)
        return res


# Para retrocompatibilidad del material en el curso
class DirectorFeatures(CrewFeatures):
    def __init__(self, min_cnt_movies=2):
        super().__init__(field='director', min_cnt_movies=min_cnt_movies)
