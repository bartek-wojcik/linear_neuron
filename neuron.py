import random
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt

load_dotenv()

delta_difference = []


def map_to_floats(array):
    return list(map(float, array))


def read_data(filename):
    inputs = []
    outputs = []
    with open(filename) as f:
        data = f.readlines()[2:]
        lines = map(str.split, data)
        for line in lines:
            inputs.append(map_to_floats(line[:-1]))
            outputs.append(float(line[-1]))
    return {
        'inputs': inputs,
        'real_outputs': outputs,
        'patterns': len(inputs[0]),
    }


def init_weights(patterns, input_weights):
    return [random.uniform(input_weights[0], input_weights[1]) for _ in range(patterns)]


def compute_output(inputs, weights):
    value = 0
    for i in range(len(weights)):
        value += weights[i] * inputs[i]
    return value


def update_weights(inputs, weights, output, real_output, learning_rate):
    for i in range(len(weights)):
        weights[i] += learning_rate * (real_output - output) * inputs[i]


def train_inputs(data, weights, learning_rate):
    for index, input in enumerate(data['inputs']):
        real_output = data['real_outputs'][index]
        output = compute_output(input, weights)
        update_weights(input, weights, output, real_output, learning_rate)


def train(epochs, data, weights, learning_rate):
    for i in range(epochs):
        train_inputs(data, weights, learning_rate)
        difference = compute_output(data['inputs'][0], weights) - data['real_outputs'][0]
        delta_difference.append(abs(difference))
    return weights


data = read_data('patterns5.txt')
epochs = int(os.getenv("EPOCHS"))
learning_rate = float(os.getenv("LEARNING_RATE"))
inter = lambda x: int(x)
input_weights = [ inter(x) for x in (os.getenv("INPUT_WEIGHS")).split(",")]
weights = init_weights(data['patterns'], input_weights)

result_weights = train(epochs, data, weights, learning_rate)

result = compute_output(data['inputs'][0], result_weights)
print(result)
print(result_weights)
plt.plot(delta_difference)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Delta', fontsize=16)
plt.show()
