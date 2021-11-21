import numpy as np
import tensorflow as tf
import tflearn


class DoubleDQNetwork:

    def __init__(self, session, dim_state, dim_action, learning_rate, tau=0.01):
        self._sess = session
        self._dim_s = dim_state
        self._dim_a = dim_action
        self._lr = learning_rate

        self._inputs = tflearn.input_data(shape=[None, self._dim_s])

        self._out, self._params = self.buildNetwork(self._inputs, 'dqn')
        self._out_target, self._params_target = self.buildNetwork(self._inputs, 'target')

        self._actions = tf.placeholder(tf.float32, [None, self._dim_a])
        self._y_values = tf.placeholder(tf.float32, [None])

        action_q_values = tf.reduce_sum(tf.multiply(self._out, self._actions), reduction_indices=1)

        self._update_target = \
            [t_p.assign(tau * g_p - (1 - tau) * t_p) for g_p, t_p in zip(self._params, self._params_target)]

        self.loss = tflearn.mean_square(self._y_values, action_q_values)
        self.optimize = tf.train.AdamOptimizer(self._lr).minimize(self.loss)

    def buildNetwork(self, state, type):
        with tf.variable_scope(type):

            # w_init = tflearn.initializations.truncated_normal(stddev=1.0)
            # weights_init=w_init,
            net = tflearn.fully_connected(state, 64, activation='relu')
            net = tflearn.fully_connected(net, 32, activation='relu')

            q_values = tflearn.fully_connected(net, self._dim_a)

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=type)
        self.trainable = tf.trainable_variables()
        return q_values, params

    def train(self, inputs, action, y_values):
        return self._sess.run([self.optimize, self.loss], feed_dict={
            self._inputs: inputs,
            self._actions: action,
            self._y_values: y_values
        })

    def predict(self, inputs):
        return self._sess.run(self._out, feed_dict={
            self._inputs: inputs,
        })

    def predict_target(self, inputs):
        return self._sess.run(self._out_target, feed_dict={
            self._inputs: inputs
        })

    def update_target(self):
        self._sess.run(self._update_target)

    def printWeights(self,nm):
        print("****************************************************")
        variables_names = [v.name for v in self.trainable]
        values = self._sess.run(variables_names)
        for v, name in zip(values,variables_names):
            name = name.replace("/","_")
            _file_name_ = str('output_w/' + str(nm) + "_" + str(name).replace(':','_') + ".txt")
            print("writing file for "+ _file_name_)
            #f = open(":\output_w\" "+nm +" _" +name + ".txt", "w")
            f = open(_file_name_, "w")
            f.write(v.__repr__())
            #print(name , " : ", (v.__repr__())