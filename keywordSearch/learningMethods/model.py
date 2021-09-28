from sklearn import linear_model
from sklearn.metrics import mean_squared_error

class GenericModel(object):
    def is_fitted(self):
        raise NotImplementedError()

    def predict(self, featurized_terms):
        return self.model.predict(featurized_terms)

    def partial_fit(self, sample_x, sample_y):
        self.model.partial_fit(sample_x, sample_y)

    def decision_function(self, diff_vec):
        return self.model.decision_function(diff_vec)

class LinearModel(GenericModel):
    def __init__(self):
        print("\t<USING LinearModel>\t")
        self.featureSize = None
        self.regularizer_weight = 0.0001
        self.learning_rate_type = 'constant'
        self.learning_rate = 0.001
        self.use_bias_term = True

        self.model = linear_model.SGDRegressor(loss='squared_loss', penalty='l2', alpha=self.regularizer_weight,
                                                   learning_rate=self.learning_rate_type, eta0=self.learning_rate,
                                                   fit_intercept=self.use_bias_term)

    def predict(self, featurized_terms):
        if self.featureSize is None:
            self.featureSize = len(featurized_terms[0])
        return self.model.predict(featurized_terms)

    def partial_fit(self, sample_x, sample_y):
        if self.is_fitted():
            loss = mean_squared_error(self.model.predict(sample_x), sample_y)
        else:
            loss = mean_squared_error([0]*len(sample_x), sample_y)
        self.model.partial_fit(sample_x, sample_y)
        return loss

    def get_weights(self):
        if self.is_fitted():
            return [w for w in self.model.coef_]
        else:
            return [0]*self.featureSize

    def is_fitted(self):
        return hasattr(self.model, 'coef_')