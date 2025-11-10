import numpy as np
import random
import math

class Network:
    """
    The Network Class that creates a Neural Network object.
    The class is built on a Linked List style structure with each layer acting as independent object 
    and the Network class being a wrapper for all.
    The class also has a very modular design as everything from NN layers to cost and activations are created as classes.
    Optimizers and Scheduler classes are soon to be implemnted as well, but for now their classes remain empty

    All arrays and inputs of any kind are assumed to numpy arrays
    """

    class Layer:
        """
        The Base class for all layer types to inherit form. Defines 2 methods that all layers must implement
        1.  feedForwad for the forward pass. Transforms input into output for regular layers such as fully connected layers,
            convolutional layers and activation layers. Performs utilities for other types of layers like reshape for flatten 
            or switch off neurons for dropout.

        2.  backpropagation for the backward pass. Transforms later errors to earlier errors for regular layers. And does backprop
            utilities for other layers.
        """

        def __init__(self, inputLayer=None):
            
            if (inputLayer != None) and (not isinstance(inputLayer, Network.Layer)):
                raise KeyError("inputLayer must be of the Layer type")
            self.inputLayer = inputLayer
            self.outputLayer = None # initialise with no child layer

            if self.inputLayer != None:
                # if the current layer is connected to a parent layer, set it as the child layer of said parent
                self.inputLayer.outputLayer = self

        def feedForward(self, x, train=False): 
            # train parameter incase the layer performance differently between training and inference e.g Dropout
            raise NotImplementedError
        
        def backpropagate(self, delta, eta):
            raise NotImplementedError

    class Activation:
        """
        The Base class for all activation function types to inherit form. Defines 3 methods that all activation functiions must implement
        1. value methods returns the output of the activation function
        2. derivative method returns the derivative of the activation function at the given input
        3. derivative_from_output method returns the derivative of the activation function at the given output,
            this is generally the most used method as it requires less computation and storage* than the derivative
            method
        
        * During backpropagtion, the derivative is to be calculated. If the derivative from output methods is used,
            It can use the activation from previous layer to calculate its values while the derivative method has to use the
            z (activation function input) from previous layer. Since the activation from the previous layer is needed to find the 
            gradient with respect to the weights anyways, the derivative method needs to keep track of one more variable.

        This format allows custom activation functions to be defined as long it inherits from the base class 
        and implements methods correctly.
        It can then be passed as an input for the ActivationLayer class
        """
        
        def __init__(self):
            pass

        def value(self, x):
            raise NotImplementedError
        
        def derivative(self, x):
            raise NotImplementedError
        
        def derivative_from_output(self, y):
            raise NotImplementedError

    class Cost:
        """
        The Base class for all cost function types to inherit form. Defines 2 methods that all cost functions must implement
        1. value to compute the cost of the network
        2. gradient to compute the gradient of the cost with respect to the output

        This format allows custom Cost functions to be defined as long it inherits from the base class 
        and implements methods correctly.
        """
        
        def __init__(self):
            pass

        def value(self, a, y):
            raise NotImplementedError
        
        def gradient(self, a, y):
            raise NotImplementedError
        
    class Optimizer:
        pass

    class Scheduler:
        pass

    class Quadratic_cost(Cost):
        """
            The Quadratic Cost implementation.
            Uses the version of the quadractic cost that is scale by 1/2 to have a clean gradient value
        """
        def __init__(self):
            pass

        def value(self, a, y):
            return np.sum(np.square(a - y)/2)

        def gradient(self, a, y):
            return (a - y)

    class CrossEntropy_cost(Cost):
        """
            The Cross Entropy Cost implementation.
            This version of the Cross Entropy Cost assumes there is no sigmoid activation explicitly defined and instead calculates from logits
            This is done for numerical stability and avoiding zero division errors like the plague.

            Since the cost is calculated from logits instead of having the form:
            -(ylna + (1-y)ln(1-a))
            where a is the activation of the last layer and thus the input to the cost function

            it instead uses 
            ln(exp(z) + 1) - yz
            where z is the output of the preceeding fully connected layer (without sigmoid activation) and thus the input 
            to the cost function
        """
        
        def __init__(self):
            pass

        def value(self, a, y):
            # np.logaddexp(a, b) is a numerically stable way to compute log(exp(a) + exp(b))
            return np.sum(np.logaddexp(0, a) - a * y)

        def gradient(self, a, y):
            # would be cleaner to use sigmoid(a) - y but sigmoid is a class not a function
            return (np.exp(-np.logaddexp(0, -a)) - y)

    class Sigmoid(Activation):
        """
            The Sigmoid (Logistic) activation.
            A numerically stable implememtation of the sigmoid activation. Traditional sigmoid implementation runs into a tricky
            problem. 
            sigmoid(x) = 1/(1 + exp(-x)) overflows for very negative values of x (as x approaches -inf) and breaks.
            sigmoid(x) = exp(x)/(exp(x) + 1) overflows for very positive value of x (as x approaches inf).

            Using the np.logaddexp is an easy way to bypass this nuisance.
        """
        
        def __init__(self):
            pass

        def value(self, x): 
            # numerically stable as regular implementation could encounter overflow for near infinite values
            return np.exp(-np.logaddexp(0, -x))

        def derivative(self, x):
            return (self.value(x)*(1 - self.value(x)))

        def derivative_from_output(self, y):
            return (y * (1 - y))
        
    class ReLU(Activation):
        """
        The ReLU (Rectified Linear Unit) activation.
        """

        def __init__(self):
            pass

        def value(self, x):
            return np.maximum(0, x)
        
        def derivative(self, x):
            return (x > 0).astype(np.float32)
        
        def derivative_from_output(self, y):
            return (y > 0).astype(np.float32)
        

    class Tanh(Activation):
        """
        The Tanh (Hyperbolic Tangent) activation.
        """

        def __init__(self):
            pass

        def value(self, x):
            return np.tanh(x)

        def derivative(self, x):
            return (1 - np.tanh(x)**2)

        def derivative_from_output(self, y):
            return (1 - y**2)
        
    class Identity(Activation):
        """
        A do nothing activation for logits output or regression.

        Created as inefficient detour to act as a quick fix until activation layers were created.
        Now are just essentially useless but adds character.
        """

        def __init__(self):
            pass

        def value(self, x):
            return x
        
        def derivative(self, x):
            return np.ones(x.shape)
        
        def derivative_from_output(self, y):
            return np.ones(y.shape)
        
    class SGD(Optimizer):
        pass

    class Momentum(Optimizer):
        pass

    class Adam(Optimizer):
        pass

    class FullyConnectedLayer(Layer):
        """
            The Fully connected layer class.
            
            The input is to be the integer value of the length of the expected input,
            the output is to be the integer value of the output (number of neurons in the layer),
            the inputLayer is a reference to the layer just before it in the Network,

            weight_initialisation is a text prompt that can be either ("large" or "small"),
            it determines the spread of the initial set of random weights.

            an example of fully connected layers of a network could be

            l1 = Network.FullyConnectedLayer(784, 128)
            l2 = Network.FullyConnectedLayer(128, 10, l1)

            A current flaw of the class is the need to specify the exact size of the input, this is usually not a problem but can
            be frustating when used immediately after a Flatten layer

            Should be fixed soon enough though

        """

        def __init__(self, input, output, inputLayer=None, weight_initialisation="small"):
            super().__init__(inputLayer)
            
            self.inputSize = input
            self.outputSize = output
            
            # bias set to have dimensions of the neurons in the layer
            self.bias = np.random.standard_normal((output, 1)) 
            
            if weight_initialisation.lower() == "small":
                self.small_weight_initialisation()
            elif weight_initialisation.lower() == "large":
                self.large_weight_initialisation()
            else:
                raise KeyError(f"{weight_initialisation} is not a valid weight initialisation.")

        def large_weight_initialisation(self): 
            # standard weight initialisation
            self.weight = np.random.standard_normal((self.outputSize, self.inputSize))

        def small_weight_initialisation(self): 
            # smarter weight initialisation which adjusts for the size of the input layer
            self.weight = np.random.standard_normal((self.outputSize, self.inputSize))/np.sqrt(self.inputSize) 

        def feedForward(self, x, train=False):
            # train parameter is completely useless since behavior is the same during training and inference


            # it's a secret tool that will help us later
            # in actuality, it uses the assumption that a backprop step was preceeded by a feedforward step
            # by storing the input of the latest feedforward step, the needed variable for backprop will be available
            self.lastInput = x 
            
            output = np.matmul(self.weight, x) + self.bias
            return output

        def backpropagate(self, delta, eta):
            # initialise the previous layer delta as None in case there is no inputLayer 
            # this staves off a bug of trying to return a value that wasn't initialised
            previous_layer_delta = None                
            
            # backpropagates the delta to find the delta of the previous layer
            if self.inputLayer != None: 
                previous_layer_delta = np.matmul(self.weight.T, delta)

            # will probably be refactored into the optimizer class soon
            # finds the grad with respect to weights and biases
            delta_weight = np.matmul(delta, self.lastInput.T)
            delta_bias = np.sum(delta, axis=1).reshape(-1, 1)
            # the bias grad is initialy of shape (n, batch_size), np.sum all the contribution of individual datapoints 

            # adjust the weight and bias accordingly
            self.weight -= eta * delta_weight
            self.bias -= eta * delta_bias

            return previous_layer_delta
            
            

    class ConvolutionalLayer(Layer):
        """
            The Convolutional layer class.

            The input is to be a tuple of the dimensions of the 2d image of the expected input,
            the input channels is to be an integer value of the number of channnels of the expected input,
            the kernel size represents the dimensions of the convolutional kernel of the layer,
            the output channels is to be an integer value of the number of channnels the layer has,

            an example of fully connected layers of a network could be

            c1 = Network.ConvolutionalLayer((28, 28), 1, (5, 5), 64)
            c2 = Network.ConvolutionalLayer((28, 28), 64, (5, 5), 64, c1)

            A current flaw of the class is that just like the fully connected layer there is a need to specify the exact size of the input,
            this is usually not a problem as long as a few basic rules are kept in mind.
            The convolutional layer as it currently exists always pads the input, has no stride parameter and does yet not support pooling.
            This means that value of "input" does not change for successive convolutional layers
            The output_channels for one convolutional layer is always the input_channels for the next

            A goal is for the Network class to become powerful and versatile enough to create various legendary neural nets like LeNet 5 and AlexNet
            Currently though, each channel in one layer are connected to every channel in the previous layer.
            Better customisability will be added later on

            Thus far, the Convolutional class has only been built and tested on the philosophy of kernels having odd number sizes for each dimensions
            so it could break if a kerenel_size of (4, 4) is used

            Finally, since there is no reverse of the Flatten class, a convolutional layer can never come after a Fully connected layer.
            It is dubious if there will be any significance to a structure that allows convolutional layers to come after fully connected ones but
            the option is sure to be added eventually.

        """
        
        def __init__(self, input, input_channels, kernel_size, output_channels, inputLayer=None):
            super().__init__(inputLayer)

            if kernel_size[0] != kernel_size[1]:
                raise KeyError("Only square kernels supported")
            if kernel_size[0] % 2 != 1:
                raise KeyError("Only odd sized kernels supported")
            self.inputSize = input

            # The kernels are 4 dimensional arrays of shape (kernel_width, kernel_height, num_of_input_channels, num_of_output_channels)
            self.kernel = np.random.standard_normal(kernel_size + (input_channels, output_channels))

            # the bias is of dimensions (output_channels, 1) since there is one shared bias for each channel
            self.bias = np.random.standard_normal((output_channels, 1)) 

        def feedForward(self, x, train=False):
            # train parameter is completely useless since behavior is the same during training and inference

            # see fully connected layer notes to understand
            self.lastInput = x  
            
            # create an output array of zeros of the correct dimensions
            output = np.zeros((x.shape[0], x.shape[1], self.kernel.shape[3], x.shape[3])) 

            # pad the input accordingly
            # the padding works by adding half of one less than the kernel size of that dimension (in zeros) to both ends of the input
            # this keeps the dimension of the ouput same as the original input 
            # but this approach clearly falls apart for even padding
            # thus even padding are currently a bug field for this class
            
            x_pad = np.zeros((x.shape[0] + 2 * (self.kernel.shape[0] // 2), x.shape[1] + 2 * (self.kernel.shape[1] // 2), x.shape[2], x.shape[3]))
            x_pad[self.kernel.shape[0] // 2 : (self.kernel.shape[0] // 2) + x.shape[0], self.kernel.shape[1] // 2 : (self.kernel.shape[1] // 2) + x.shape[1], :, :] = x
            x = x_pad 
            # assign the padded input as the input

            # loop through all elements in the output and calculate their values individually
            # Not quite because instead of also looping through each channel in the output,
            # all values on a certain coordinate in every channel (and every data point in the batch) is calculated at once, 
            # since they are all calculated from the same section of the input
            # to do this, the section of the input data that maps to that section of the output array is found and stored.
            # then through simple but careful numpy multiplication it is multplied by the entire kernel to find the output element
            # The last step is to sum over a number of dimensions as required
            # The rule of thumb is that if a convolutional process is implemented using basic multiplication operation instead of matmul,
            # there will need to be a use (or many uses) of np.sum to adjust the dimensions
            
            for h in range(output.shape[1]):
                for w in range(output.shape[0]):
                    sub_x = x[w : w + self.kernel.shape[0] , h : h + self.kernel.shape[1], : , :] # grab the section of the input data
                    inter =  sub_x.reshape((sub_x.shape) + (1,)) * self.kernel.reshape((self.kernel.shape[0], self.kernel.shape[1], self.kernel.shape[2], 1, self.kernel.shape[3]))
                    # reshape a couple of things to prep for the multiplication 
                    output[w][h][:, :] = np.sum(np.sum(np.sum(inter, axis=0), axis=0), axis=0).T # the transpose is used to retain the original order

            # add the bias as usual
            output =  output + self.bias 
            
            return output

        def backpropagate(self, delta, eta):
            # initialise the previous layer delta as None in case there is no inputLayer 
            # (this staves off a bug of trying to return a value that wasn't initialised)
            previous_layer_delta = None 
            
            if self.inputLayer != None: 
                # backpropagates the delta to find the delta of the previous layer
                # just like in fully connected layers, backpropagation remains like feedforward in reverse
                # The only real difference is that while fully connected layers used the transpose of the weight matrix,
                # convolutional layers use the 180 degree rotation of the kernel.
                # this is really interesting because technically speaking, only the backpropagation part of the implementation actually constitutes a convolution
                # Another way to think of it though will be that the true kernel is a 180 degree rotation of what we traditional call the kernel and that...
                # ... the feedforward is the true convolution while the backpropagation is just the regular fake convolution with the true kernel.
                # I do wonder what the deeper mathematical connection between these ideas and connections is though
                # Basically everything else remains the same including the padding of the delta just like with feedforward
                
                previous_layer_delta = np.zeroes(delta.shape)
                kernel = np.transpose(np.rot90(self.kernel, 2), (0, 1, 3, 2))
                
                delta_pad = np.zeros((delta.shape[0] + 2 * (kernel.shape[0] // 2), delta.shape[1] + 2 * (kernel.shape[1] // 2), delta.shape[2], delta.shape[3]))
                delta_pad[kernel.shape[0] // 2 : (kernel.shape[0] // 2) + delta.shape[0], kernel.shape[1] // 2 : (kernel.shape[1] // 2) + delta.shape[1], :, :] = delta

                for h in range(previous_layer_delta.shape[1]):
                    for w in range(previous_layer_delta.shape[0]):
                        sub_delta = delta_pad[w : w + kernel.shape[0] , h : h + kernel.shape[1], : , :]
                        inter =  sub_delta.reshape((sub_delta.shape) + (1,)) * kernel.reshape((kernel.shape[0], kernel.shape[1], kernel.shape[2], 1, kernel.shape[3]))
                        previous_layer_delta[w][h][:, :] = np.sum(np.sum(np.sum(inter, axis=0), axis=0), axis=0).T

            # initialise the gradient with respect to the weights as a matrix of zeros
            # it has one more dimension than the kernel
            # the last dimension is the number of data points in the batch (batch_size)
            # similar to the feedforward calculation, a section of the padded input data is extracted.
            # this is the section that is corresponds to all values of a given coordinate on the delta
            # both matrices are reshaped appropriately and multiplied to give the resulting weight gradient
            # since the weights are shared for all neurons, the results for all coordinates are added together
            # the resulting value is summed over the last axis to combine all data in the batch
            
            delta_weight = np.zeros(self.kernel.shape + (delta.shape[3],))
            
            input_pad = np.zeros((self.lastInput.shape[0] + 2 * (self.kernel.shape[0] // 2), self.lastInput.shape[1] + 2 * (self.kernel.shape[1] // 2), self.lastInput.shape[2], self.lastInput.shape[3]))
            input_pad[self.kernel.shape[0] // 2 : (self.kernel.shape[0] // 2) + self.lastInput.shape[0], self.kernel.shape[1] // 2 : (self.kernel.shape[1] // 2) + self.lastInput.shape[1], :, :] = self.lastInput
       

            for h in range(delta.shape[1]):
                for w in range(delta.shape[0]):
                    delta_weight += input_pad[w : w + self.kernel.shape[0], h : h + self.kernel.shape[1], :, :].reshape((self.kernel.shape[0], self.kernel.shape[1], input_pad.shape[2], 1, input_pad.shape[3])) * delta[w][h][:, :].reshape(1, 1, 1, delta.shape[2], delta.shape[3])
       
            delta_weight = np.sum(delta_weight, axis=-1)
            
            delta_bias = np.sum(np.sum(np.sum(delta, axis=0), axis=0), axis=1).reshape(-1, 1)
            # the bias gradient is found by summing through all the delta values for all coordinate points an finally for all data in the batch 

            self.kernel -= eta * delta_weight
            self.bias -= eta * delta_bias

            return previous_layer_delta
        
    class ActivationLayer(Layer):
        
        def __init__(self, inputLayer=None, activation=None):
            super().__init__(inputLayer)

            # checks first as not to unknowningly try to call None.lower()
            if activation == None: 
                self.activation = Network.Identity()
            
            # checks for the case of activation chosen using strings
            elif isinstance(activation, str):
                # strings must correspond to available type activations   
                if activation.lower() == "sigmoid":
                    self.activation = Network.Sigmoid()
                elif activation.lower() == "relu":
                    self.activation = Network.ReLU()
                elif activation.lower() == "tanh":
                    self.activation = Network.Tanh()
                else:
                    raise KeyError(f"{activation} is not a valid activation.")
                
            # checks if the activation is a valid object type instead
            elif isinstance(activation, Network.Activation):
                self.activation = activation

            else:
                raise KeyError(f"{activation} is not a valid activation.")

        def feedForward(self, x, train=False):
            # saves lastInput for computing backpropagation
            self.lastInput = x
            return self.activation.value(x)
        
        def backpropagate(self, delta, eta):
            # we used derivative instead of derivative from output this is because self.lastInput is a "z" not an "a"
            return (delta * self.activation.derivative(self.lastInput)) 

    class Dropout(Layer):
        """"
        "1D" dropout for fully connected layers and convolutional layers as well. Drops out individual neurons regardless of 
        if the input is a fully connected layer or a convolutional layer. 

        Implements the modern version of dropout which adjusts the value during training and leaves inference normal.
        Has the advantage of being able to completely discard the dropout layer after training and 
        pretending nothing ever happend.
        """

        def __init__(self, inputLayer, p=0.5):

            super().__init__(inputLayer)
            self.mask = None
            self.p = p

        def feedForward(self, x, train=False):
            # during training neurons are masked out randomly bur left as is during inference
            if train:
                # the mask has the same dimensions as the input except for the last dimension (batch) which is one
                # this is important as it avoids using different masks for different data points in the same batch
                # the input is scaled up by the 1/(1-p) to ensure the expected value of the layer remains as during inference
                self.mask = np.expand_dims(np.random.binomial(1, 1 - self.p, size=x.shape[:-1]).astype(np.float32), axis=-1)
                return (self.mask * x) / (1 - self.p)
            else:
                return x
        
        def backpropagate(self, delta, eta):
            # backpass is very similar to forward pass
            return (self.mask * delta) / (1 - self.p)

    class Flatten(Layer):
        """
            Flatten class
            
            It acts to flatten a convolutional layer's output which is 4d to a 2d form that can be used by fully connnected layers.
        """

        def __init__(self, inputLayer=None):
            
            super().__init__(inputLayer)

            self.lastInput = None

        def feedForward(self, x, train=False):
            # useless train parameter as expected
            # feedforward only reshapes the input to expected form       
            self.lastInput = x
            return np.reshape(x, (-1, x.shape[-1]))

        def backpropagate(self, delta, eta):
            # the backpropagation only reshapes the "flat" delta back to a 4d form
            return np.reshape(delta, self.lastInput.shape)
                            
    @staticmethod
    def computeOutputSize(layer):
        # computes the Output size of a layer
        # currently has no use but could prove valuable in the future
        if isinstance(layer, Network.FullyConnectedLayer):
            return (layer.outputSize, )
        elif isinstance(layer, Network.ConvolutionalLayer):
            return layer.inputSize + (layer.kernel.shape[-1],)
        elif isinstance(layer, Network.ActivationLayer):
            return (Network.computeOutputSize(layer.inputLayer)) 
        else:
            raise KeyError(f"{layer} does not support computeOutputSize")

    def __init__(self, conv=False):
        # currently a quick fix to use the conv parameter to show that it is a conv net as some methods act differently depending on what form of NN it is 
        self.firstLayer = None
        self.lastLayer = None
        self.conv = conv

    def compile(self, firstLayer):
        # could potentially be removed from being called explicitly by incorporating into the __init__ method
        # compiles the Network by looping through the network and setting the first and layers to appropriate values
        self.firstLayer = firstLayer
        walk = firstLayer
        
        while walk.outputLayer != None:
            walk = walk.outputLayer

        self.lastLayer = walk
    
    def predict(self, x, train=False):
        # gives the network output on a given input by looping through layers and running feedforwards
        a = x
        walk = self.firstLayer
        
        while walk != None:
            a = walk.feedForward(a, train)
            walk = walk.outputLayer
        return a

    def train(self, X_train, y_train, X_test, y_test, eta, mini_batch_size, epochs, cost="cross entropy", track_training_metrics=False):
        """
            Starts a training session

            expected parameters include the X and y values for the training and test data

            the eta is the learning rate

            mini_batch_size specifies how many data points should be evaluated before update to weights

            epochs specify the amount of training cycles through the entire training data

            cost could be a text prompt that specifies what cost will be used
            e.g 
            "cross entropy"
            "quadratic"

            or a entry of the specific cost type needed
            e.g
            Network.CrossEntropy_cost()
            Network.Quadratic_cost()

            track_training_metrics determines if the training metrics are to be evaluated after every epoch or just the test will do

            finally the metrics are returned at the end of the training in the form
            ((training_costs, training_accuracies) , (test_costs, test_accuracies))

            A simple showcase of what this Network in action could look like is 

            cnn1 = Network.ConvolutionalLayer((28, 28), 1, (3, 3), 8)
            act1 = Network.ActivationLayer(cnn1, Network.Sigmoid())
            flat1 = Network.Flatten(act1)
            fcl1 = Network.FullyConnectedLayer(28 * 28 * 8, 128, flat1)
            act2 = Network.ActivationLayer(fcl1, Network.Sigmoid())
            drop1 = Network.Dropout(act2, p=0.2)
            fcl2 = Network.FullyConnectedLayer(128, 10, drop1)

            net = Network()
            net.compile(cnn1)

            net.train(X_train, y_train, X_test, y_test, 0.05, 20, 50)

            NB: In the above case, the network doesn't use a final activation layer and returns logits since it uses the 
            default cross entropy cost during training.
                        
        """

        # check if it is a string
        if isinstance(cost, str):
            if cost.lower() == "cross entropy":
                self.cost = Network.CrossEntropy_cost()
            elif cost.lower() == "quadratic":
                self.cost = Network.Quadratic_cost()
            else:
                raise KeyError(f"{cost} is not a valid Cost function.")
            
        elif isinstance(cost, Network.Activation):
            if isinstance(cost, Network.CrossEntropy_cost):
                self.cost = cost

        else:
                raise KeyError(f"{cost} is not a valid Cost function.")
           
        training_accuracies, training_costs = [], []
        test_accuracies, test_costs = [], []

        training_size = y_train.shape[1]
        for i in range(epochs):
            # at the start of every epoch new training and test data clones are created and shuffled
            
            current_X_train = X_train[:, :]
            current_y_train = y_train[:, :]

            # the data is transposed inorder to keep the length of the data in front
            # the transpose is returned after shuffling but it is needed in order to shuffle about the correct axis
            # convolutional data is of a different shape to regular data and is thus handled differently
            if self.conv:
                current_X_train = np.transpose(current_X_train, (3, 0, 1, 2))
            else:
                current_X_train = np.transpose(current_X_train, (1, 0))
            current_y_train = np.transpose(current_y_train)

            # a seed is chosen at random and this seed is then used to ensure the X and y training data are shuffled identically
            seed = random.randint(1, 100000)
            np.random.seed(seed)
            np.random.shuffle(current_X_train)
            np.random.seed(seed)
            np.random.shuffle(current_y_train)

            if self.conv:
                current_X_train = np.transpose(current_X_train, (1, 2, 3, 0))
            else:
                current_X_train = np.transpose(current_X_train, (1, 0))
            current_y_train = np.transpose(current_y_train)
                
            
            for b in range(math.ceil(training_size/mini_batch_size)):
                # the batches are divided and fit one after the other
                # convolutional data is once again handled separately
                if not self.conv:
                    self.fit(current_X_train[:, b * mini_batch_size : (b + 1) * mini_batch_size], current_y_train[:, b * mini_batch_size : (b + 1) * mini_batch_size], eta)
                else:
                    self.fit(current_X_train[:, :, :, b * mini_batch_size : (b + 1) * mini_batch_size], current_y_train[:, b * mini_batch_size : (b + 1) * mini_batch_size], eta)
                
            print(f"Epoch {i}")

            if track_training_metrics:
                # calculate and keep track of training metrics if needed
                training_accuracy = self.check_accuracy(X_train, y_train)
                training_cost = self.calculate_cost(X_train, y_train)

                training_accuracies.append(training_accuracy)
                training_costs.append(training_cost)

                print(f"Training Accuracy: {training_accuracy} out of {y_train.shape[1]}")
                print(f"Training Cost: {training_cost} ")

                
            test_accuracy = self.check_accuracy(X_test, y_test)
            test_cost = self.calculate_cost(X_test, y_test)

            test_accuracies.append(test_accuracy)
            test_costs.append(test_cost)

            print(f"Test Accuracy: {test_accuracy} out of {y_test.shape[1]}")
            print(f"Test Cost: {test_cost}")
            
        return((training_costs, training_accuracies) , (test_costs, test_accuracies))
        
    def fit(self, X, y, eta):
        # to fit, one forward pass is taken IMMEDIATELY followed by a backward pass which starts with the delta calculated using the output
        
        a = self.predict(X, True)
        delta = self.cost.gradient(a, y)
        
        walk = self.lastLayer
        
        while walk != None:
            delta = walk.backpropagate(delta, eta)
            walk = walk.inputLayer
    

    def check_accuracy(self, X, y):
        # the accuracy could be checked in a much faster fashion
        #    result = np.argmax(self.predict(X), axis=0)
        #    solution = np.argmax(y, axis=0)
        #    return np.sum(result == solution)
        #    result works even when logits not probabilities are used since argmax works regardless
        # This above version will work but large enough datasets working on Networks with large enough layers (i.e a fully connected layer with a 
        # large "input_size * output_size" - which can occur after a Flatten layer) can run into Memory errors using the first implementation
        # The below implementation basically calculates the accuracy in chunks and completes the dataset in 100 chunks
        # this 100 is arbitary and a temporary fix that runs into # problems
        # if the entire dataset is less than 100, it will throw an error
        # if the dataset is not divisible by 100, weird stuff could happen 
        # if the dataset is large enough it will still throw a Memory error
   
        total = 0
        for i in range(100):
            if self.conv:
                result = np.argmax(self.predict(X[:, :, :, i*(X.shape[-1]//100) : (i+1)*(X.shape[-1]//100)]), axis=0)
            else:
                result = np.argmax(self.predict(X[:, i*(X.shape[-1]//100) : (i+1)*(X.shape[-1]//100)]), axis=0)
            solution = np.argmax(y[:, i*(X.shape[-1]//100) : (i+1)*(X.shape[-1]//100)], axis=0)
            total += np.sum(result == solution)
        return total

    def calculate_cost(self, X, y):
        # the cost could be checked in a much faster fashion
        #    a = self.predict(X)
        #    return self.cost.value(a, y)
        # the caveats of the check_accuracy method all apply      
        cost_value = 0
        for i in range(100):
            if self.conv:
                a = self.predict(X[:, :, :, i*(X.shape[-1]//100) : (i+1)*(X.shape[-1]//100)])
            else:
                a = self.predict(X[:, i*(X.shape[-1]//100) : (i+1)*(X.shape[-1]//100)])
            cost_value += self.cost.value(a, y[:, i*(X.shape[-1]//100) : (i+1)*(X.shape[-1]//100)])/y.shape[1]
        return cost_value
            
# Final thoughts
# In addition to the issues listed throughout the inline comments,
# there are still lacking base classes for Optimizers, Schedulers and many more
# Layers with multiple types like convolutional and dropout are getting away with using the group name for the popular form for now
# 1d and 3d conv net as well as 2d and 3d dropout will be implemented in the future causing a number of name changes
# Having subsequent layers infer redundant parameters will be useful
# specification of sizes may need to go
# sequential like method that automatically bundles layers and thus doesn't need explicit calling of previous layers could be useful
# Not needing to call compile explicitly may need to go
# It could also help to put trailer layers in the Network class as attributes as this is the more standard doubly-linked list format and can definitely help