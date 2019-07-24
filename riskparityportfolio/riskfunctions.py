import tensorflow as tf

class RiskConcentrationFunction:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def evaluate(self):
        return tf.reduce_sum(tf.square(self.risk_concentration_vector()))

    # the vector g in Feng & Palomar 2015
    def risk_concentration_vector(self):
        raise NotImplementedError("this method should be implemented in the child class")

    # jacobian of the vector function risk_concentration_vector with respect to weights
    def jacobian_risk_concentration_vector(self):
        raise NotImplementedError("this method should be implemented in the child class")
        # I'm not gonna use autograd here. We have the derivatives analytically already.
        #with tf.GradientTape() as t:
        #    t.watch(self.portfolio.weights)
        #    risk_vec = self.risk_concentration_vector()
        #return t.jacobian(risk_vec, self.portfolio.weights)


class RiskContribOverBudgetDoubleIndex(RiskConcentrationFunction):
    def risk_concentration_vector(self):
        N = len(self.portfolio.weights)
        marginal_risk = tf.math.multiply(self.portfolio.weights,
                tf.linalg.matvec(self.portfolio.covariance, self.portfolio.weights))
        normalized_marginal_risk = tf.math.divide(marginal_risk, self.portfolio.budget)
        return tf.tile(normalized_marginal_risk, [N]) - repeat(normalized_marginal_risk, N)


class RiskContribOverVarianceMinusBudget(RiskConcentrationFunction):
    def risk_concentration_vector(self):
        marginal_risk = tf.math.multiply(self.portfolio.weights,
                tf.linalg.matvec(self.portfolio.covariance, self.portfolio.weights))
        return marginal_risk / tf.reduce_sum(marginal_risk) - self.portfolio.budget

    def jacobian_risk_concentration_vector(self):
        Sigma_w = tf.linalg.matvec(self.portfolio.covariance, self.portfolio.weights)
        r = tf.multiply(self.portfolio.weights, Sigma_w)
        sum_r = tf.reduce_sum(r)
        Ut = tf.linalg.diag(Sigma_w) + tf.multiply(self.portfolio.covariance, self.portfolio.weights)
        return Ut / sum_r - 2 / (sum_r ** 2) * (r[..., None] * Sigma_w[..., None])

def repeat(vector, times):
    vector = tf.reshape(tf.tile(tf.reshape(vector, [-1, 1]), [1, times]), [-1])
    return vector

