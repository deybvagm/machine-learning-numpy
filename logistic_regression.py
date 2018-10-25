from utils import load_dataset, plot_cost_over_time
from lr_utils import initialize_parameters, forward_propagation, \
    backward_propagation, compute_cost, update_parameters, predict, calculate_accuracy


def train_algorithm(X, Y, num_iterations=1000, learning_rate=0.01):
    num_px = X.shape[0]

    parameters = initialize_parameters(num_px)

    costs = []

    for i in range(num_iterations):
        A = forward_propagation(parameters, X)
        cost = compute_cost(A, Y)
        grads = backward_propagation(X, A, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        costs.append(cost)
        if i % 100 == 0:
            print('cost after iteration {}:{}'.format(i, cost))

    plot_cost_over_time(costs)
    return parameters


train_X, train_Y, test_X, test_Y, classes = load_dataset()
params = train_algorithm(train_X, train_Y)

train_pred_Y = predict(params, train_X)
print('Train accuracy: ')
calculate_accuracy(train_Y, train_pred_Y)
test_pred_Y = predict(params, test_X)
print('Test accuracy: ')
calculate_accuracy(test_Y, test_pred_Y)
