import numpy as np
from tensorflow.keras import datasets
from Neural_Net import Network

(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

def vectorise(x):
    blank = [0 for i in range(10)]
    new_blank = np.array(blank).reshape((10, 1))
    new_blank[x][0] = 1
    return new_blank

mnist_train = [(x.reshape((784, 1))/255, vectorise(y)) for x, y in zip(x_train, y_train)]
mnist_test = [(x.reshape((784, 1))/255, vectorise(y)) for x, y in zip(x_test, y_test)]


mnist_net_1 = Network((784, 100, 10), "large")
print(mnist_net_1.check_accuracy(mnist_test))
history_1 = mnist_net_1.train(mnist_train, mnist_test, 0.5, 0.0, 50, 100, 0.5, cost="quadratic", track_training_metrics=True)

# mnist_net_1.save("fashion_mnist_100_300_000_50_100_large_quadractic_1.nn")


# with open("Neural Net Performance.txt", "a") as nn_file:
#     nn_file.write("\n")
#     nn_file.write("fashion_mnist_100_300_000_50_100_large_quadractic_1\n")
#     nn_file.write(str(history_1))




# mnist_net_2 = Network((784, 100, 10), "large")
# print(mnist_net_2.check_accuracy(mnist_test))
# history_2 = mnist_net_2.train(mnist_train, mnist_test, 3.0, 0.0, 50, 100, cost="quadratic", track_training_metrics=True)
# mnist_net_2.save("fashion_mnist_100_300_000_50_100_large_quadractic_2.nn")

# with open("Neural Net Performance.txt", "a") as nn_file:
#     nn_file.write("\n")
#     nn_file.write("fashion_mnist_100_300_000_50_100_large_quadractic_2\n")
#     nn_file.write(str(history_2))


# mnist_net_3 = Network((784, 100, 10), "large")
# print(mnist_net_3.check_accuracy(mnist_test))
# history_3 = mnist_net_3.train(mnist_train, mnist_test, 3.0, 0.0, 50, 100, cost="quadratic", track_training_metrics=True)
# mnist_net_3.save("fashion_mnist_100_300_000_50_100_large_quadractic_3.nn")

# with open("Neural Net Performance.txt", "a") as nn_file:
#     nn_file.write("\n")
#     nn_file.write("fashion_mnist_100_300_000_50_100_large_quadractic_3\n")
#     nn_file.write(str(history_3))

# # fashion_mnist_net_2 = Network((784, 100, 10))

# # print(fashion_mnist_net_2.check_accuracy(fashion_mnist_test))
# # fashion_mnist_net_2.train(fashion_mnist_train, fashion_mnist_test, 0.2, 50, 100)
# # fashion_mnist_net_2.save("fashion_mnist_100_2.nn")

# # fashion_mnist_net_3 = Network((784, 100, 10))

# # print(fashion_mnist_net_3.check_accuracy(fashion_mnist_test))
# # fashion_mnist_net_3.train(fashion_mnist_train, fashion_mnist_test, 0.2, 50, 100)
# # fashion_mnist_net_3.save("fashion_mnist_100_3.nn")