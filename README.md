# MNIST mechanistic interpretability

A learning notebook that trains linear classifiers and small convolutional networks on MNIST, then explores what the trained models use to make predictions.

The notebook base is built from PyTorch's tutorial (https://pytorch.org/tutorials/beginner/nn_tutorial.html) by Jeremy Howard/fast.ai. Then, I extended these simple neural nets to try some common mechanistic interpretability ablation techniques. For instance, in CNNs turning channels on and off to see how validation loss moves or seeing how sensitive our classification is to small changes in pixels.
