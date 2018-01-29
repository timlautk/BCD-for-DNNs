# Backprop Experiments using Keras
Vanilla SGD:
MNIST Dataset Test Accuracy after 100 epochs: 94.63% (Residual Connection; mean squared loss), 94.63% (MLP; mean squared loss), 97.08% (MLP; cross-entropy loss) 

CIFAR-10 Dataset Test Accuracy after 100 epochs: 46.36% (Residual Connection; mean squared loss), 47.03% (MLP; mean squared loss), 54.5% (MLP; cross-entropy loss) 

Other optimizers such as RMSprop, Adagrad, Adadelta, Adam, Adamax, Nesterov Adam, or those available in native TensorFlow can also be used for experiments. (Left for future work)
