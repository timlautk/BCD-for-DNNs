# Backprop Experiments using Keras
Vanilla SGD:

MNIST Dataset Test Accuracy after 100 epochs: ~95% (Residual Connection; mean squared loss), ~95% (MLP; mean squared loss), ~98% (Residual Connection; hinge loss), ~98% (MLP; hinge loss)  

CIFAR-10 Dataset Test Accuracy after 100 epochs: ~49% (Residual Connection; mean squared loss), ~47% (MLP; mean squared loss), ~51% (Residual Connection; hinge loss), ~42% (Residual Connection; hinge loss) 

Other optimizers such as RMSprop, Adagrad, Adadelta, Adam, Adamax, Nesterov Adam, or those available in native TensorFlow can also be used for experiments. (Left for future work)
