def activation(net):
    if net > 0:
        return 1
    else:
        return -1


def train(inputs, outputs, weights, threshold):
    numInputs = len(inputs[0])
    numPatterns = len(inputs)
    for i in range(numPatterns):
        net = 0
        for j in range(numInputs):
            net += weights[j] * inputs[i][j]
        for j in range(numInputs):
            weights[j] += inputs[i][j] * outputs[i]
        print("Weights:", weights[0], weights[1])
        threshold -= outputs[i]
        print("Shold:", threshold)


def train1(inputss, outputss, weightss, thresholds):
    numInputs = len(inputss[0])
    numPatterns = len(inputss)
    for i in range(numPatterns):
        net = 0
        for j in range(numInputs):
            net += weightss[j] * inputss[i][j]
        for j in range(numInputs):
            weightss[j] += inputss[i][j] | outputss[i]
        print("Weights:", weightss[0], weightss[1])
        thresholds -= outputss[i]
        print("Shold:", thresholds)


inputs = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
outputs = [-1, -1, -1, 1]
weights = [0, 0]
threshold = 0
train(inputs, outputs, weights, threshold)
for i in range(len(inputs)):
    net = 0
    for j in range(len(inputs[i])):
        net += weights[j] * inputs[i][j]
    print("net:", net)
    result = activation(net + threshold)
    print("Input: (", inputs[i][0], ",", inputs[i][1], ") Output:", result)


inputss = [[-1, -1], [1, -1], [-1, 1], [1, 1]]
outputss = [-1, -1, -1, 1]
weightss = [0, 0]
thresholds = 0
train1(inputss, outputss, weightss, thresholds)
for i in range(len(inputss)):
    net = 0
    for j in range(len(inputs[i])):
        net += weightss[j] * inputss[i][j]
    print("net:", net)
    result = activation(net - thresholds)
    print("Input: (", inputss[i][0], ",", inputss[i][1], ") Output:", result)


