import numpy as np
import random

class nusa:
    def __init__(self, layernumber, input_data, training_data, weightarray=None, biasarray=None):
        self.layernumber = layernumber
        self.input = input_data
        self.training_data = training_data
        self.weightarray = weightarray if weightarray is not None else []
        self.biasarray = biasarray if biasarray is not None else []

    def begin(self):
        layer1 = self.inputlayer()
        layer2 = self.layer(10, layer1)
        layer3 = self.layer(10, layer2)
        layer4 = self.layer(10, layer3)
        layer5 = self.layer(10, layer4)
        layer6 = self.layer(10, layer5)
        layer7 = self.layer(10, layer6)
        layer8 = self.layer(12, layer7)
        output = self.outputlayer(layer8)
        print("Final Output:", output)

    def inputlayer(self):
        output = np.array([])
        for i in range(len(self.input)):
            output = np.append(output, self.neuron(input=self.input[i], passn=1))
        return output

    def layer(self, neuron_number, input_data):
        output = np.array([])
        if len(input_data) < neuron_number:
            x = neuron_number - len(input_data)
            for i in range(x):
                input_data = np.append(input_data, i)
        for i in range(neuron_number):
            output = np.append(output, self.neuron(input=input_data[i], passn=1))
        return output

    def neuron(self, input, passn):
        if passn >= 1:
            weight = random.random() * 0.8548741055637802
            bias = random.random()
            self.weightarray.append(weight)
            self.biasarray.append(bias)
        else:
            weight = self.weightarray[-1] if self.weightarray else 0.5
            bias = self.biasarray[-1] if self.biasarray else 0.5
        transfer = (weight * input) + bias
        transfer = self.sigmoid(transfer)
        return transfer

    def outputlayer(self, input_data):
        array = input_data
        array_size = len(array)
        slice1 = array[:array_size//2]
        slice2 = array[array_size//4:array_size//4*3]
        slice3 = array[array_size//2:]
        dot_product_1 = np.dot(slice1, slice1)
        dot_product_2 = np.dot(slice2, slice2)
        dot_product_3 = np.dot(slice3, slice3)
        dot_products_array = np.array([dot_product_1, dot_product_2, dot_product_3])
        output = np.array([])
        output = np.append(output, self.neuron(input=dot_product_1, passn=1))
        output = np.append(output, self.neuron(input=dot_product_2, passn=1))
        output = np.append(output, self.neuron(input=dot_product_3, passn=1))
        output = self.softmax(dot_products_array)
        return output

    def optimizer(self):
        # Placeholder for optimizer implementation
        pass

    def loss(self, y_true, y_pred):
        epsilon = 1e-15  # Small value to avoid division by zero
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip values to avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
        return exp_x / np.sum(exp_x)

# Example usage
inputarr = np.array([1, 2, 3, 4, 5])
training_data = np.array([0, 1, 0])  # Placeholder for training data
cow = nusa(5, inputarr, training_data)
cow.begin()
