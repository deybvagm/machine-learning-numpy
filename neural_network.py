from utils import load_dataset, plot_cost_over_time
from nn_utils import initialize_parameters, forward_propagation, backward_propagation, \
    compute_cost, update_parameters, predict, accuracy


def train_neural_network(X, Y, num_iterations=4000, learning_rate=0.01):
    n_x = X.shape[0]
    n_h = 7
    n_y = 1

    parameters = initialize_parameters(n_x, n_h, n_y)
    costs = []

    for i in range(num_iterations):
        A2, cache = forward_propagation(parameters, X)
        cost = compute_cost(A2, Y)
        costs.append(cost)
        gradients = backward_propagation(X, Y, cache)
        parameters = update_parameters(parameters, gradients, learning_rate)

        if i % 100 == 0:
            print('Cost after iteration {}:{}'.format(i, cost))

    plot_cost_over_time(costs)
    return parameters


# Data
train_X, train_Y, test_X, test_Y, classes = load_dataset()

# Training
model = train_neural_network(train_X, train_Y)

# Predictions with model
train_pred_Y = predict(train_X, model)
print('Accuracy on training set: {}'.format(accuracy(train_Y, train_pred_Y)))
test_pred_Y = predict(test_X, model)
print('Accuracy on test set: {}'.format(accuracy(test_Y, test_pred_Y)))
