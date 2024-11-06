# MR & N

This is the **MR & N** repo, which stands for Math, Research, and Numpy.

## Objective

My goal with this is to make a Neural Network Module from scratch,  
without using any library which is not Numpy.  
Starting from simple fully connected layers to Convolutional Layers,  
and maybe eventually transformers.  
TensorFlow, PyTorch, and Theano better not get too comfortable because a new underdog is in town.

## There are 2 files in the repo:

### Legacy

The first is the `Neural_Net` file, and essentially serves as the **legacy form** of the project.

- **Backend**  
  Its basic data structure is based on an array.

- **Functionality**  
  It implements a number of features like momentum, learning rate schedule, and dropout but is fairly slow and has no support for convolutional neural networks.

- **Thoughts and Influences**  
  It serves as a showcase of my level of expertise when I started the project, as a lot of my Neural Net from scratch knowledge was from Michael Nielson's book, and that is quite evident in the broad strokes of the code.

---

### Future

The second is the `New_Neural_Net` file.  
It is the more modern and newer version of the project.

- **Backend**  
  It replaces the array-based structure and replaces it with a doubly linked list.  
  It is also very generous with its use of classes.

- **Functionality**  
  It is more modular, upgradeable, efficient, better commented, and in my opinion, more intuitive.  
  Given its relatively newness, it doesn't yet have most of the functionality of the first, and its error catching is less than stellar.  
  These are very short-term problems though and will be addressed very soon.  
  All that being said, it has a successful implementation of Convolutional Neural Networks,  
  and is more than likely the only one that will continue receiving support into the future.

- **Thoughts and Influences**  
  This version is not completely original either, although its influence is more spread across various books and papers.  
  The largest single influence on it probably is the striking similarity its initialization has with Keras.  
  *(To be the King, you sometimes have to learn from the king)*