import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLU = ReLULayer()
        self.layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        self.params()
        for value in self.params().values():
            value.grad[...] = 0

        X_out_layer1 = self.layer1.forward(X)
        X_out_ReLU = self.ReLU.forward(X_out_layer1)
        X_out_layer2 = self.layer2.forward(X_out_ReLU)

        loss1_reg, grad_loss1_reg = l2_regularization(self.layer1.W.value, self.reg)
        loss2_reg, grad_loss2_reg = l2_regularization(self.layer2.W.value, self.reg)

        loss_pred, dloss = softmax_with_cross_entropy(X_out_layer2, y)

        loss = loss_pred + loss1_reg + loss2_reg

        dlayer2 = self.layer2.backward(dloss)
        self.layer2.W.grad += grad_loss2_reg
        dReLU = self.ReLU.backward(dlayer2)
        self.layer1.W.grad += grad_loss1_reg
        dlayer1 = self.layer1.backward(dReLU)

        self.params()

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

        pred = np.zeros(X.shape[0], np.int)

        X_out_layer1 = self.layer1.forward(X)
        X_out_ReLU = self.ReLU.forward(X_out_layer1)
        X_out_layer2 = self.layer2.forward(X_out_ReLU)

        # proba = np.apply_along_axis(softmax, axis=1, arr=X_out_layer2)
        # pred = np.argmax(proba, axis=1)

        pred = np.argmax(softmax(X_out_layer2), axis=1)

        return pred

    def params(self):
        result = \
            {
                "W1": self.layer1.params()["W"],
                "B1": self.layer1.params()["B"],
                "W2": self.layer2.params()["W"],
                "B2": self.layer2.params()["B"]
            }

        return result
