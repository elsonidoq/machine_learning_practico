from . import transformers
from sklearn.pipeline import make_union, make_pipeline
from sklearn.feature_extraction import DictVectorizer


def get_features_pipe(
        use_years: bool, use_genre: bool,
        use_director: bool, director_kws: dict = None, post_processing=None):
    steps = []
    if use_years:
        steps.append(make_pipeline(transformers.YearsAgo(), DictVectorizer(sparse=False)))

    if use_genre:
        steps.append(make_pipeline(transformers.GenreDummies(), DictVectorizer(sparse=False)))

    if use_director:
        director_kws = director_kws or {}
        # cuando hacemos **director_kws usamos ese diccionario para pasar parametros
        steps.append(make_pipeline(transformers.DirectorFeatures(**director_kws), DictVectorizer(sparse=False)))

    res = make_union(*steps)
    if post_processing:
        res = make_pipeline(res, post_processing)
    return res


def get_model_pipe(features_pipe, model):
    return make_pipeline(features_pipe, model)