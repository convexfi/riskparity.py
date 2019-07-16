import tensorflow as tf

class RiskConcentrationFunction:
    def evaluate(self, weights=None, covariance=None, budget=None):
        return(tf.square(tf.reduce_sum(self.risk_concentration_vector(weights=weights,
                                                            covariance=covariance,
                                                            budget=budget))))
    # the vector g in Feng & Palomar 2015
    def risk_concentration_vector(self, weights=None, covariance=None,
                                  budget=None):
        raise NotImplementedError("this method should be implemented in the child class")

    # gradient of the vector function risk_concentration_vector with respect to weights
    def gradient_risk_concentration_vector(self, weights=None, covariance=None,
                                           budget=None):
        with tf.GradientTape() as t:
            tf.watch(weights)
            return t.gradient(self.risk_concentration_vector, weights)

class RiskContribOverBudgetDoubleIndex(RiskConcentrationFunction):
    def risk_concentration_vector(self, weights=None, covariance=None, budget=None):
        N = len(weights)
        marginal_risk = tf.math.multiply(weights, tf.linalg.matvec(covariance, weights))
        normalized_marginal_risk = tf.math.divide(marginal_risk, budget)
        return tf.tile(normalized_marginal_risk, [N]) - repeat(normalized_marginal_risk, N)


def repeat(vector, times):
    vector = tf.reshape(tf.tile(tf.reshape(vector, [-1, 1]), [1, times]), [-1])
    return vector

