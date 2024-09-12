import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)


from ..api.base import ExplainerMixin, ExplanationMixin
from ..api.templates import FeatureValueExplanation
from ..utils._explanation import (
    gen_name_from_class,
    gen_local_selector,
    gen_global_selector,
    gen_perf_dicts,
)
from sklearn.base import ClassifierMixin, is_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.utils.validation import check_is_fitted
from abc import ABC, abstractmethod

from ..utils._clean_x import preclean_X

import logging

class NaiveBayesExplanation(ExplanationMixin):
    """Explanation object specific to Naive Bayes."""

    explanation_type = None

    def __init__(
        self,
        explanation_type,
        internal_obj,
        feature_names=None,
        feature_types=None,
        name=None,
        selector=None,
    ):
        """Initializes class.

        Args:
            explanation_type:  Type of explanation.
            internal_obj: A jsonable object that backs the explanation.
            feature_names: List of feature names.
            feature_types: List of feature types.
            name: User-defined name of explanation.
            selector: A dataframe whose indices correspond to explanation entries.
        """

        self.explanation_type = explanation_type
        self._internal_obj = internal_obj
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.name = name
        self.selector = selector

    def data(self, key=None):
        """Provides specific explanation data.

        Args:
            key: A number/string that references a specific data item.

        Returns:
            A serializable dictionary.
        """
        if key is None:
            return self._internal_obj["overall"]
        return self._internal_obj["specific"][key]

    def visualize(self, key=None):
        """Provides interactive visualizations.

        Args:
            key: Either a scalar or list
                that indexes the internal object for sub-plotting.
                If an overall visualization is requested, pass None.

        Returns:
            A Plotly figure.
        """
        from interpret.visual.plot import (
            sort_take,
            mli_sort_take,
            get_sort_indexes,
            get_explanation_index,
            plot_horizontal_bar,
            mli_plot_horizontal_bar,
        )

        if isinstance(key, tuple) and len(key) == 2:
            provider, key = key
            if (
                "mli" == provider
                and "mli" in self.data(-1)
                and self.explanation_type == "global"
            ):
                explanation_list = self.data(-1)["mli"]
                explanation_index = get_explanation_index(
                    explanation_list, "global_feature_importance"
                )
                scores = explanation_list[explanation_index]["value"]["scores"]
                sort_indexes = get_sort_indexes(
                    scores, sort_fn=lambda x: -abs(x), top_n=15
                )
                sorted_scores = mli_sort_take(
                    scores, sort_indexes, reverse_results=True
                )
                sorted_names = mli_sort_take(
                    self.feature_names, sort_indexes, reverse_results=True
                )
                return mli_plot_horizontal_bar(
                    sorted_scores,
                    sorted_names,
                    title="Overall Importance:<br>Coefficients",
                )
            else:  # pragma: no cover
                raise RuntimeError("Visual provider {} not supported".format(provider))
        else:
            data_dict = self.data(key)
            if data_dict is None:
                return None

            if self.explanation_type == "global" and key is None:
                data_dict = sort_take(
                    data_dict, sort_fn=lambda x: -abs(x), top_n=15, reverse_results=True
                )
                figure = plot_horizontal_bar(
                    data_dict, title="Overall Importance:<br>Coefficients"
                )
                return figure
        return super().visualize(key)


class NaiveBayesClassifier(ExplainerMixin, ClassifierMixin):
    """Naive Bayes Classifier."""

    available_explanations = ["global", "local"]
    explainer_type = "model"

    def __init__(self, feature_names=None, feature_types=None, **kwargs):
        """Initializes Naive Bayes classifier.

        Args:
            feature_names: List of feature names.
            feature_types: List of feature types.
            **kwargs: Kwargs sent to __init__() method of Naive Bayes.
        """
        self.feature_names = feature_names
        self.feature_types = feature_types
        self.feature_types_in_ = feature_types 
        self.kwargs = kwargs

    def _model(self):
        return self.sk_model_

    def fit(self, X, y):
        """Fits model to provided instances.

        Args:
            X: Numpy array for training instances.
            y: Numpy array as training labels.

        Returns:
            Itself.
        """
        self.X = X
        self.y = y 
        self.sk_model_ = GaussianNB(**self.kwargs)
        self.sk_model_.fit(X, y)
        self.feature_names_in_ = self.feature_names
        self.classes_ = self.sk_model_.classes_
        self.has_fitted_ = True
        return self

        self.n_features_in_ = len(self.feature_names_in_)
        if is_classifier(self):
            self.classes_ = model.classes_

        self.X_mins_ = np.min(X, axis=0)
        self.X_maxs_ = np.max(X, axis=0)
        self.categorical_uniq_ = {}

        for i, feature_type in enumerate(self.feature_types_in_):
            if feature_type == "nominal" or feature_type == "ordinal":
                self.categorical_uniq_[i] = list(sorted(set(X[:, i])))

        unique_val_counts = np.zeros(len(self.feature_names_in_), dtype=np.int64)
        for col_idx in range(len(self.feature_names_in_)):
            X_col = X[:, col_idx]
            unique_val_counts.itemset(col_idx, len(np.unique(X_col)))

        self.global_selector_ = gen_global_selector(
            len(self.feature_names_in_),
            self.feature_names_in_,
            self.feature_types_in_,
            unique_val_counts,
            None,
        )

    
    def predict(self, X):
        """Predicts on provided instances.

        Args:
            X: Numpy array for instances.

        Returns:
            Predicted class label per instance.
        """

        check_is_fitted(self, "has_fitted_")

         #X, _, _ = unify_data(
        #    X, n_samples, self.feature_names_in_, self.feature_types_in_, True, 0
        #)

        X, n_samples = preclean_X(X, self.feature_names_in_, self.feature_types_in_)

    
        return self._model().predict(X)


    #def _calculate_feature_contributions(self, instance):
    #    """Calcula la contribución de cada característica a la probabilidad de la clase predicha."""
    #     log_prob_x_given_y = self.sk_model_._joint_log_likelihood(instance.reshape(1, -1))
    #    class_prior = np.log(self.sk_model_.class_prior_)
    #    log_likelihood = log_prob_x_given_y + class_prior
    #    class_prob = np.exp(log_likelihood)
    #    feature_contributions = log_prob_x_given_y[0]  # Contribuciones de cada característica
    #    return feature_contributions


    def feature_weights_naive_bayes_local(self, X, X_train, y_train):
        """
        Calcula las probabilidades de cada clase para una observación dada, utilizando un modelo Naive Bayes Gaussiano entrenado con los datos proporcionados.

        Args:
            X: Una observación individual (array NumPy o lista).
            X_train: Datos de entrenamiento (array NumPy o DataFrame de Pandas).
            y_train: Etiquetas de clase de los datos de entrenamiento (array NumPy o lista).

        Returns:
            Un diccionario con las probabilidades de cada clase para la observación dada.
        """

        if isinstance(X_train, pd.DataFrame):
            X_train = X.values  # Convertir a array de NumPy si es necesario

        classes = np.unique(y_train)
        n_features = X_train.shape[1]
        weights = np.zeros(n_features)
        intercept = 0

        for c in classes:
            class_mask = (y_train == c)
            class_prior = np.mean(class_mask)
            other_prior = 1 - class_prior

        # Evitar división por cero en el cálculo del intercepto
        if other_prior == 0:
            other_prior = 1e-20  # Pequeño valor para evitar la división por cero

        intercept = np.log(class_prior / other_prior)


        for i in range(n_features):
            mean_c = np.mean(X_train[class_mask, i])
            std_c = np.std(X_train[class_mask, i])

            mean_other = np.mean(X_train[~class_mask, i])
            std_other = np.std(X_train[~class_mask, i])

                # Evitar divisiones por cero y resultados infinitos
            epsilon = 1e-20  # Pequeño valor para evitar la división por cero
            std_c = max(std_c, epsilon)
            std_other = max(std_other, epsilon)

            p_xi_given_c = 1 / (np.sqrt(2 * np.pi) * std_c) * np.exp(-((X[i] - mean_c) ** 2) / (2 * std_c ** 2))
            p_xi_given_other = 1 / (np.sqrt(2 * np.pi) * std_other) * np.exp(-((X[i] - mean_other) ** 2) / (2 * std_other ** 2))

            # Evitar logaritmos de cero
            if p_xi_given_c == 0:
                p_xi_given_c= epsilon
            if p_xi_given_other == 0:
                p_xi_given_other= epsilon
        
            weights[i] += (np.log(p_xi_given_c / p_xi_given_other))

        return intercept, weights


    def feature_weights_naive_bayes_global(self,X, y):

        if isinstance(X, pd.DataFrame):
            X = X.values  # Convertir a array de NumPy si es necesario

        classes = np.unique(y)
        n_features = X.shape[1]
        weights = np.zeros(n_features)
        intercept = 0

        for c in classes:
            class_mask = (y == c)
            class_prior = np.mean(class_mask)
            other_prior = 1 - class_prior

        # Evitar división por cero en el cálculo del intercepto
        if other_prior == 0:
            other_prior = 1e-20  # Pequeño valor para evitar la división por cero

        intercept += np.log(class_prior / other_prior)

        for i in range(n_features):
            mean_c = np.mean(X[class_mask, i])
            std_c = np.std(X[class_mask, i])

            mean_other = np.mean(X[~class_mask, i])
            std_other = np.std(X[~class_mask, i])

            # Evitar divisiones por cero y resultados infinitos
            epsilon = 1e-20  # Pequeño valor para evitar la división por cero
            std_c = max(std_c, epsilon)
            std_other = max(std_other, epsilon)

            p_xi_given_c = 1 / (np.sqrt(2 * np.pi) * std_c) * np.exp(-((X[:, i] - mean_c) ** 2) / (2 * std_c ** 2))
            p_xi_given_other = 1 / (np.sqrt(2 * np.pi) * std_other) * np.exp(-((X[:, i] - mean_other) ** 2) / (2 * std_other ** 2))

            # Evitar logaritmos de cero
            p_xi_given_c[p_xi_given_c == 0] = epsilon
            p_xi_given_other[p_xi_given_other == 0] = epsilon

            weights[i] += np.sum(np.log(p_xi_given_c / p_xi_given_other))/len(X[:, i])

        return intercept, weights
    

    def explain_local(self, X, y, name=None):
        """Provides local explanations for provided instances using Naive Bayes.

        Args:
            X: Numpy array for X to explain.
            y: Numpy vector for y to explain.
            name: User-defined explanation name.

        Returns:
            An explanation object, visualizing feature-value pairs
            for each instance as horizontal bar charts.
        """
        check_is_fitted(self, "has_fitted_")

        classes = np.unique(self.classes_)

        if name is None:
            name = gen_name_from_class(self)

        # Preprocesamiento de datos
        #X, _, _ = unify_data(
        #    X, len(X), self.feature_names_in_, self.feature_types_in_, False, 0
        #)

        # Obtener probabilidades de clase y etiquetas predichas
        model = self._model()  # Obtiene el modelo Naive Bayes subyacente
        predictions = model.predict_proba(X)
        predicted_classes = np.argmax(predictions, axis=1)

        data_dicts = []
        for i, instance in enumerate(X):
            # Calcular la contribución de cada característica a la probabilidad de la clase predicha
            feature_contributions = self.feature_weights_naive_bayes_local(instance, self.X,self.y)

            data_dict = {
                "data_type": "univariate",
                "names": self.feature_names_in_,
                "intercept": feature_contributions[0],
                "scores": feature_contributions[1],
                "values": instance,
                "Positive boundary target":classes[1],
                "Negative boundary target":classes[0],
                "decision boundary": (feature_contributions[0]+np.sum(feature_contributions[1])), 
                "perf": {
                    "probability": predictions[i].tolist(),
                    "class": self.classes_[predicted_classes[i]],
                    "predicted_score": predictions[i][predicted_classes[i]],
                    "actual_score": y[i] if y is not None else None, 
                    "actual": y[i] if y is not None else None, 
                    "predicted": self.classes_[predicted_classes[i]]
                }
            }
            data_dicts.append(data_dict)

        # Generar el selector local
        selector = gen_local_selector(data_dicts, is_classification=True)

    # Crear y devolver la explicación
        return FeatureValueExplanation(
            "local",
            {"specific": data_dicts},
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=selector,
        )


    def explain_global( self, name=None):
        """Provides global explanation for Naive Bayes model."""
        check_is_fitted(self, "has_fitted_")

        if name is None:
            name = gen_name_from_class(self)

        model = self._model()


        feature_contributions = self.feature_weights_naive_bayes_global(self.X ,self.y)

        # Create dictionaries to hold information
        overall_data_dict = {
                "names": self.feature_names_in_,
                "scores": feature_contributions[1],
                "extra": {"names": ["Intercept"], "scores": [feature_contributions[0]]},
                "value": {"feature_weigths": feature_contributions[0]}
                }


        # Assemble the internal object
        internal_obj = {
            "overall": overall_data_dict,
            "nb": [
                {
                    "explanation_type": "global_feature_importance",
                    "value": {"feature_weigths": feature_contributions[0]},
                }
            ],
        }

        # Create a global selector for visualization
        unique_val_counts = np.ones(len(self.feature_names_in_), dtype=np.int64) * len(self.classes_)  # One value per feature per class
        global_selector = gen_global_selector(
            len(self.feature_names_in_),
            self.feature_names_in_,
            self.feature_types_in_,
            unique_val_counts,
            model,
        )

        return FeatureValueExplanation(  # Use your custom explanation class
            "global",
            internal_obj,
            feature_names=self.feature_names_in_,
            feature_types=self.feature_types_in_,
            name=name,
            selector=global_selector,
        )

