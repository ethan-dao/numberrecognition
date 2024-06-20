# Number recognition

# Building the neural network
Using the MNIST handwriting dataset in Tensorflow, I created a feed forward neural network that analyzes handwritten digits correctly ~96% of the time. First, I separated the data provded by Tensorflow as a dataset into training and testing sets, and normalized each data point by dividing it by 255. From there, I coverted the labels to one-hot encoding and built my model. For my model, I took in the input as a 28x28 pixel image of a handwritten digit, and flattened it into a 784 element vector. From there, I used the ReLU activation function and a dropout of 0.2 for each layer to end up with a layer of 10 neurons, which would be the output of my model. From there, I got the softmax prediction for my output, turning the logits into probabilities, and trained my model for 5 epochs. For the loss function of my model, I used a categorical cross entropy, as we are doing image classification, and I used the Adam optimizer in this case instead of SGD, measuring the accuracy of the image classification. 

# Training and testing the neural network
Now, I trained this model for 5 epochs with early stopping as a callback so that it the training would stop when there was a decrease in accuracy. Next, I tested the model by evaluating the model on the testing data, and received an accuracy of around 96% after 5 epochs!
