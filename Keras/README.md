# Backprop Experiments using Keras
Vanilla SGD:

MNIST Dataset Test Accuracy after 100 epochs: ~94.6% (Residual Connection; mean squared loss), ~94.6% (MLP; mean squared loss), ~97% (MLP; cross-entropy loss) 

CIFAR-10 Dataset Test Accuracy after 100 epochs: ~46% (Residual Connection; mean squared loss), ~47% (MLP; mean squared loss), ~51% (MLP; cross-entropy loss) 

Other optimizers such as RMSprop, Adagrad, Adadelta, Adam, Adamax, Nesterov Adam, or those available in native TensorFlow can also be used for experiments. (Left for future work)
