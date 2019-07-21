import tensorflow as tf

class RiskConcentrationFunction:
    def __init__(self, weights, covariance, budget):
        self.weights = weights
        self.covariance = covariance
        self.budget = budget

    def evaluate(self):
        return tf.square(tf.reduce_sum(self.risk_concentration_vector()))
    # the vector g in Feng & Palomar 2015
    def risk_concentration_vector(self):
        raise NotImplementedError("this method should be implemented in the child class")

    # jacobian of the vector function risk_concentration_vector with respect to weights
    def jacobian_risk_concentration_vector(self):
        with tf.GradientTape() as t:
            t.watch(self.weights)
            risk_vec = self.risk_concentration_vector()
        return t.jacobian(risk_vec, self.weights)

class RiskContribOverBudgetDoubleIndex(RiskConcentrationFunction):
    def risk_concentration_vector(self):
        N = len(self.weights)
        marginal_risk = tf.math.multiply(self.weights, tf.linalg.matvec(self.covariance, self.weights))
        normalized_marginal_risk = tf.math.divide(marginal_risk, self.budget)
        return tf.tile(normalized_marginal_risk, [N]) - repeat(normalized_marginal_risk, N)


def repeat(vector, times):
    vector = tf.reshape(tf.tile(tf.reshape(vector, [-1, 1]), [1, times]), [-1])
    return vector

