# Codigo distilado de la notebook clase-1/03_problem_framing_and_feature_engineering.ipynb

import numpy as np
from pathlib import Path
import pandas as pd


def load_title_basics(path: Path):
    title_basics = pd.read_csv(path / 'title.basics.tsv', sep='\t')

    def parse_genres(genres):
        if isinstance(genres, float) or genres == r'\N':
            return ['no-genre']
        else:
            return genres.split(',')

    # Convertimos runtimeMinutes a float. No se puede tener una columna de tipo int con NaN
    title_basics.runtimeMinutes = (
        title_basics.runtimeMinutes.apply(lambda x: np.nan if not x.isdigit() else x).astype(float)
    )

    title_basics['genres'] = title_basics.genres.apply(parse_genres)
    title_basics['startYear'] = title_basics.startYear.apply(lambda x: np.nan if x == r'\N' else int(x))

    title_basics = title_basics[
        # Dejamos tvSpecial, video y tvMovie por ahora, vamos a ver de que se tratan
        ~title_basics.titleType.isin(['tvEpisode', 'tvSeries', 'tvMiniSeries', 'videoGame', 'tvShort', 'short'])
        # Que tengan valor de runtimeMinutes
        & ~title_basics.runtimeMinutes.isna()
        # Menos de 3 horas y media para no descartar a titanic
        & (title_basics.runtimeMinutes <= 3.5 * 60)
        # Descartamos los shorts
        & title_basics.genres.apply(lambda x: 'Short' not in x)

    ]
    return title_basics


def load_title_ratings(path: Path):
    return pd.read_csv(path / 'title.ratings.tsv', sep='\t')


def load_movie_directors(path: Path):
    # Me quedo solo con los que fueron directores
    principals_df = pd.read_csv(path / 'title.principals.tsv', sep='\t')

    movies_directors = principals_df[principals_df.category == 'director'].copy()
    # Calculo un ranking por pelicula segun el ordering
    movies_directors['director_rank'] = (
        movies_directors.sort_values('ordering')
            .groupby('tconst')
            .cumcount()
    )

    # Me quedo con el "director principal" por pelicula
    movies_directors = movies_directors[movies_directors.director_rank == 0]

    # Me quedo solo con la columna del director
    movies_directors = (
        movies_directors.rename(columns={'nconst': 'director'})
        [['tconst', 'director']]
    )

    return movies_directors


def load_data(path: Path):
    # Sacado del trabajo hecho en clase-1
    print("Loading title basics...")
    title_basics = load_title_basics(path)
    print("Loading title ratings...")
    title_ratings = load_title_ratings(path)
    print("Loading movie directors...")
    movie_directors = load_movie_directors(path)

    print("Merging everything...")
    movies = (
        title_basics.merge(title_ratings, on='tconst') # descartamos aquellas que no tienen rating
                    .merge(movie_directors, on='tconst', how='left')
    )

    movies = movies[~movies.averageRating.isna()].copy()

    return movies


def load_rating_train_dev_test(movies: pd.DataFrame, train_max_year=2015, dev_max_year=2017, sample_count: int = None):
    """
    :param movies: Movies dataframe
    :param train_max_year: cut year for training
    :param dev_max_year: cut year for dev (and starts test)
    :param sample_count: whether to take a sample (useful for testing the code). Ignored when it is None
    """
    if sample_count:
        movies = movies.sample(sample_count)

    train_df = movies[movies.startYear <= train_max_year]
    dev_df = movies[(movies.startYear > train_max_year) & (movies.startYear <= dev_max_year)]
    test_df = movies[movies.startYear > dev_max_year]

    X_train = train_df.to_dict(orient='records')
    X_dev = dev_df.to_dict(orient='records')
    X_test = test_df.to_dict(orient='records')

    y_train = train_df.averageRating.values
    y_dev = dev_df.averageRating.values
    y_test = test_df.averageRating.values

    return dict(X_train=X_train, y_train=y_train, X_dev=X_dev, y_dev=y_dev, X_test=X_test, y_test=y_test)
