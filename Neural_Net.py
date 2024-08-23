import numpy as np
import random
import pickle

class Network:

    class Quadratic_cost:
        
        def __init__(self):
            pass

        def value(self, a, y):
            return np.sum(np.square(a - y)/2)

        def gradient(self, a, y):
            return (a - y)

    class CrossEntropy_cost:

        def __init__(self):
            pass

        def value(self, a, y):
            return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a)))

        def gradient(self, a, y):
            return (-y/a + (1-y)/(1-a))
        
    def sigmoid(x):
        return 1/(1 + np.exp(-x))
        
    def sigmoid_prime(x):
        return (Network.sigmoid(x)*(1 - Network.sigmoid(x)))
         
    def __init__(self, size, weight_initialisation="small"):
        self.size = size
        if weight_initialisation.lower() == "small":
            self.small_weight_initialisation()
        elif weight_initialisation.lower() == "large":
            self.large_weight_initialisation()
        else:
            raise KeyError(f"{weight_initialisation} is not a valid weight initialisation.")

    def large_weight_initialisation(self):
        self.biases = [np.random.standard_normal((self.size[i], 1)) for i in range(1, len(self.size))]
        self.weights = [np.random.standard_normal((self.size[i], self.size[i-1])) for i in range(1, len(self.size))]   

    def small_weight_initialisation(self):
        self.biases = [np.random.standard_normal((self.size[i], 1)) for i in range(1, len(self.size))]
        self.weights = [np.random.standard_normal((self.size[i], self.size[i-1]))/np.sqrt(self.size[i-1]) for i in range(1, len(self.size))]
        
    def feedforward(self, x):
        a = x
        for i in range(len(self.size)-1):
            z = np.matmul(self.weights[i], a) + self.biases[i]
            a = Network.sigmoid(z)
        return a

    def train(self, training_data, test_data, eta, mini_batch_size, epochs, cost="cross entropy"):
        if cost.lower() == "cross entropy":
            self.cost = Network.CrossEntropy_cost()
        elif cost.lower() == "quadratic":
            self.cost = Network.Quadratic_cost()
        else:
            raise KeyError(f"{cost} is not a valid weight initialisation.")
        
        for i in range(epochs):
            new_training_data = training_data[:]
            random.shuffle(new_training_data)
            mini_batches = [new_training_data[i*mini_batch_size: (i+1)*mini_batch_size] for i in range(len(new_training_data)//mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_weights(mini_batch, eta)
                
            accuracy = self.check_accuracy(test_data)
        
            print(f"Epoch {i}: {accuracy} out of {len(test_data)}")

    def update_weights(self, mini_batch, eta):
        cumulative_delta_weights = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        cumulative_delta_biases = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))]

        for data in mini_batch:
            new_delta_weights, new_delta_biases = self.backpropagate(data)
            cumulative_delta_weights = [cdw + ndw for cdw, ndw in zip(cumulative_delta_weights, new_delta_weights)]
            cumulative_delta_biases = [cdb + ndb for cdb, ndb in zip(cumulative_delta_biases, new_delta_biases)]

        self.weights = [w - (eta/len(mini_batch))*cdw for w, cdw in zip(self.weights, cumulative_delta_weights)]
        self.biases = [b - (eta/len(mini_batch))*cdb for b, cdb in zip(self.biases, cumulative_delta_biases)]

    def backpropagate(self, data):
        X, y = data
        delta_weights = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        delta_biases = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))]
        a = X
        a_s = [X]
        deltas = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))]
        z_s = []
        for i in range(len(self.size) - 1):
            z = np.matmul(self.weights[i], a) + self.biases[i]
            z_s.append(z)
            a = Network.sigmoid(z)
            a_s.append(a)

        delta = np.multiply(self.cost.gradient(a_s[-1], y), Network.sigmoid_prime(z))
        deltas[-1] = delta

        delta_weights[-1] = np.matmul(delta, np.transpose(a_s[-2]))
        delta_biases[-1] = delta


        
        for l in range(len(self.size) - 3, -1, -1):
            delta = np.multiply(np.matmul(np.transpose(self.weights[l+1]), deltas[l+1]), Network.sigmoid_prime(z_s[l]))
            deltas[l] = delta

            delta_weights[l] = np.matmul(delta, np.transpose(a_s[l]))
            delta_biases[l] = delta

        return delta_weights, delta_biases

    def check_accuracy(self, test_data):
        total = 0
        for X, y in test_data:
            result = np.argmax(self.feedforward(X))
            solution = np.argmax(y)
            if result == solution:
                total += 1
        return total

    def save(self, file_path):
        with open(file_path,'wb') as f:
            pickle.dump(self, f)
        f.close()

    def load(file_path):
        with open(file_path,'rb') as f:
            model = pickle.load(f)
        f.close()
        return model 