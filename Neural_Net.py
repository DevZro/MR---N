import numpy as np
import random
import pickle
import math

class Network: 
    """Neural Network class, produces neural net objects for classification."""
    class Quadratic_cost:
        """ Quadratic cost class that can compute cost value and gradient for back propagation
        PS: all functions take vector inputs"""
        def __init__(self):
            pass

        def value(self, a, y):
            return np.sum(np.square(a - y)/2)

        def gradient(self, a, y):
            return (a - y)

    class CrossEntropy_cost:
        """ CrossEntropy cost class that can compute cost value and gradient for back propagation
        PS: all functions take vector inputs"""
        def __init__(self):
            pass

        def value(self, a, y):
            return np.sum(np.nan_to_num(-y*np.log(a) - (1-y)*np.log(1-a))) # nan to num to handle special case of 0*log(0)

        def gradient(self, a, y):
            return (-y/a + (1-y)/(1-a))
        
    def exponential(eta, epoch, exponential_constant):
        return eta*((exponential_constant)**epoch)

    def sigmoid(x):
        return 1/(1 + np.exp(-x)) 
        
    def sigmoid_prime(x):
        return (Network.sigmoid(x)*(1 - Network.sigmoid(x)))
         
    def __init__(self, size, weight_initialisation="small"):
        self.size = size # stores received size tuple as an attribute

        """if elif else block to handle weight_initialisation inputs"""
        if weight_initialisation.lower() == "small":
            self.small_weight_initialisation()
        elif weight_initialisation.lower() == "large":
            self.large_weight_initialisation()
        else:
            raise KeyError(f"{weight_initialisation} is not a valid weight initialisation.") # any input that is not "small" or "large" is invalid and throws and exception
        
        self.weight_velocity = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        self.bias_velocity = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))] # velocity attributes are created to help with momentum based training

    def large_weight_initialisation(self):
        self.biases = [np.random.standard_normal((self.size[i], 1)) for i in range(1, len(self.size))] # for each layer (hidden or final) a bias vector of layer size lenght is created
        self.weights = [np.random.standard_normal((self.size[i], self.size[i-1])) for i in range(1, len(self.size))] # for each pair of adjacent layers, a weight matrix of layer sizes dimensions is created, convention puts later layer as row

    def small_weight_initialisation(self):
        """scales the value of the weights by the root of the size of the previous layer"""
        self.biases = [np.random.standard_normal((self.size[i], 1)) for i in range(1, len(self.size))]
        self.weights = [np.random.standard_normal((self.size[i], self.size[i-1]))/np.sqrt(self.size[i-1]) for i in range(1, len(self.size))]
        
    def feedforward(self, x):
        """ Compute the output of the network for a given input"""
        a = x
        for i in range(len(self.size)-1):
            z = np.matmul(self.weights[i], a) + self.biases[i] 
            a = Network.sigmoid(z)
        return a

    def train(self, training_data, test_data, eta, lmbda, mini_batch_size, epochs, alpha=0.0, cost="cross entropy",  track_training_metrics=False, learning_rate_schedule="exponential", learning_rate_constant=1.0):
        """
        initialises training of network. Requires paramters such as training data, test data, learning rate, lamdba (L2 regularisation constant),
        alpha (momentum parameter),  batch size, epochs, cost and boolean indicating whether or not to track training data metrics.

        There is a cost parameter which sets the cost object.
        The cost can either be a recognised keyword such as quadratic or an object of a custom class which can be supported.
        Custom classes must support both the value and gradient methods.
        The validity of the gradient method and the custom class as a whole is left to the creator and will not be checked by code.

        It also contains learning rate schedule and learning rate constant parameters.
        The learning rate schedule can either be a recognised keyword such as exponential or a custom function which can be supported.
        Custom functions must take 3 arguments (eta, current_epoch(counting from 0), learning rate constant) to work.
        Since custom functions are completely ... custom, the use of the constant is entirely up to the creator.
        The standard learning rate schedule is initialised with and exponential function with constant set to 1 to replicate a constant function

        The training follows a L2 regularised form of the momentum based stochastic gradient descent algorithm, but they can be reduced to the basic SGD
        case by setting both lmbda and alpha to zero
        ps: for the sake of modularity, cost will be made to accept any object with value and gradient methods.
        """

        """
        cost attribute is set
        """

        if ("value" in dir(cost)) and ("gradient" in dir(cost)): #checks if the cost (potenially custom) is of a compatible form
            self.cost = cost
        elif cost.lower() == "cross entropy":
            self.cost = Network.CrossEntropy_cost()
        elif cost.lower() == "quadratic":
            self.cost = Network.Quadratic_cost()
        else:
            raise KeyError(f"{cost} is not a valid cost initialisation.")
        
        """
        learning rate schedule is set
        """
        if ("function" in type(learning_rate_schedule)) or ("method" in type(learning_rate_constant)):
            pass
        elif learning_rate_schedule.lower() == "exponential":
            learning_rate_schedule = Network.exponential
        else:
            raise KeyError(f"{learning_rate_schedule} is not a valid learning rate schedule initialisation.")
        



        
        training_accuracies, training_costs = [], []
        test_accuracies, test_costs = [], []
        
        training_size = len(training_data)


        for i in range(epochs): # data is shuffled and split into batches to be trained
            new_training_data = training_data[:] # new data is created so as to preserve order after each suffle
            random.shuffle(new_training_data)
            mini_batches = [new_training_data[i*mini_batch_size: (i+1)*mini_batch_size] for i in range(math.ceil(len(new_training_data)/mini_batch_size))]
            for mini_batch in mini_batches:
                self.update_weights(mini_batch, learning_rate_schedule(eta, i, learning_rate_constant), lmbda, alpha, training_size) # weights and biases are adjusted using each mini batch
                
            print(f"Epoch {i}")

            """
            block to track relevant metrics
            """
            if track_training_metrics:
                training_accuracy = self.check_accuracy(training_data)
                training_cost = self.calculate_cost(training_data)

                training_accuracies.append(training_accuracy)
                training_costs.append(training_cost)

                print(f"Training Accuracy: {training_accuracy} out of {len(training_data)}")
                print(f"Training Cost: {training_cost} ")

                
            test_accuracy = self.check_accuracy(test_data)
            test_cost = self.calculate_cost(test_data)

            test_accuracies.append(test_accuracy)
            test_costs.append(test_cost)

            print(f"Test Accuracy: {test_accuracy} out of {len(test_data)}")
            print(f"Test Cost: {test_cost}")

        return ((training_accuracies, training_costs), (test_accuracies, test_costs)) # returns the history of training and test metrics for each epoch

    def update_weights(self,mini_batch, eta, lmbda, alpha, training_size):
        cumulative_delta_weights = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        cumulative_delta_biases = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))] # change to biases and weights to be added initialised as zeros

        for data in mini_batch: # for each data point in the mini batch, cost gradients and gotten and added to a running total
            new_delta_weights, new_delta_biases = self.backpropagate(data)
            cumulative_delta_weights = [cdw + ndw for cdw, ndw in zip(cumulative_delta_weights, new_delta_weights)]
            cumulative_delta_biases = [cdb + ndb for cdb, ndb in zip(cumulative_delta_biases, new_delta_biases)]
    
        self.weight_velocity = [alpha*wv + (eta/len(mini_batch))*cdw for wv, cdw in zip(self.weight_velocity, cumulative_delta_weights)] 
        self.bias_velocity = [alpha*bv + (eta/len(mini_batch))*cdb for bv, cdb in zip(self.bias_velocity, cumulative_delta_biases)]

        self.weights = [w*(1-(eta*lmbda)/training_size) - wv for w, wv in zip(self.weights, self.weight_velocity)]
        self.biases = [b - bv for b, bv in zip(self.biases, self.bias_velocity)] # weights and biases and updated using the average of the total gradient, momentum and L2 regularisation

    def backpropagate(self, data):
        X, y = data
        delta_weights = [np.zeros_like(self.weights[i]) for i in range(len(self.weights))]
        delta_biases = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))] # gradients lists are initialised with zeros
        a = X
        a_s = [X] # activations list is created
        deltas = [np.zeros_like(self.biases[i]) for i in range(len(self.biases))] # errors list is initialised with zeros
        z_s = [] # zs list is created
        for i in range(len(self.size) - 1): # activations and zs lists are filled
            z = np.matmul(self.weights[i], a) + self.biases[i]
            z_s.append(z)
            a = Network.sigmoid(z)
            a_s.append(a)

        delta = np.multiply(self.cost.gradient(a_s[-1], y), Network.sigmoid_prime(z))
        deltas[-1] = delta # final layer error is calculated and appended into list

        delta_weights[-1] = np.matmul(delta, np.transpose(a_s[-2]))
        delta_biases[-1] = delta # gradient lists are filled alongside error list


        
        for l in range(len(self.size) - 3, -1, -1):
            """
            l layer is to start at the second to last layer since the last has been calculated 
            len - 3 because ...
            1. Python counts from 0 so last layer will be len - 1
            2. We start at second to last layer since last layer has been taken care of
            3. errors list excludes input layer since only hidden and output layers can have errors
            """
            delta = np.multiply(np.matmul(np.transpose(self.weights[l+1]), deltas[l+1]), Network.sigmoid_prime(z_s[l]))
            deltas[l] = delta

            delta_weights[l] = np.matmul(delta, np.transpose(a_s[l]))
            delta_biases[l] = delta

        return delta_weights, delta_biases

    def check_accuracy(self, test_data): # check and count up how many of a given data, a network can classify correctly
        total = 0
        for X, y in test_data:
            result = np.argmax(self.feedforward(X))
            solution = np.argmax(y)
            if result == solution:
                total += 1
        return total

    def save(self, file_path): # save model using pickle
        with open(file_path,'wb') as f:
            pickle.dump(self, f)
        f.close()

    def load(file_path): # load model using pickle
        with open(file_path,'rb') as f:
            model = pickle.load(f)
        f.close()
        return model 
    
    def calculate_cost(self, data): # evaluate average cost of a data set
        total = 0
        for X, y in data:
            total += self.cost.value(self.feedforward(X), y)
        return total/len(data)