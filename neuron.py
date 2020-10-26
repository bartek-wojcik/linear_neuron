import random
from dotenv import load_dotenv
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter(indent=4)


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

def result_table(weights):
    headers = [ "w" + str(x) for x in range (1,len(weights) + 1) ]
    row = []
    for weight in weights:
        row.append(round(weight, 3))
    rows = [row]
    return tabulate( rows, headers, tablefmt="latex")


data = read_data('patterns5.txt')
epochs = int(os.getenv("EPOCHS"))
learning_rate = float(os.getenv("LEARNING_RATE"))
inter = lambda x: int(x)
input_weights_range = [ inter(x) for x in (os.getenv("INPUT_WEIGHS")).split(",")]
weights = init_weights(data['patterns'], input_weights_range)

result_weights = train(epochs, data, weights, learning_rate)
computed_outputs = [compute_output(pattern, result_weights) for pattern in data['inputs']]

results = [ f"Result for {index+1}. pattern: " + str(c_output) + f". Diffrence with real is: {abs(c_output - data['real_outputs'][index]) }" for index, c_output in enumerate(computed_outputs)]

print("After training weights given below are used to compute output")
print(result_weights)
pp.pprint(results)

plt.plot(delta_difference)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Delta', fontsize=16)
plt.show()
