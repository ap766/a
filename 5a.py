import numpy as np
from functools import reduce

def perceptron(weight, bias, x):
    model = np.add(np.dot(x, weight), bias)
    print('model: {}'.format(model))
    logit = 1/(1+np.exp(-model))
    print('Type: {}'.format(logit))
    return np.round(logit)

def compute(logictype, weightdict, dataset):
    weights = np.array([ weightdict[logictype][w] for w in weightdict[logictype].keys()])
    output = np.array([ perceptron(weights, weightdict['bias'][logictype], val) for val in dataset])
    print(logictype)
    return logictype, output


def main():
    logic = {
        'logic_and' : {
            'w0': -0.1,
            'w1': 0.2,
            'w2': 0.2
        },

        'logic_nand': {
            'w0': 0.6,
            'w1': -0.8,
            'w2': -0.8
        },

        'bias': {
            'logic_and': -0.2,
            'logic_nand': 0.3,

        }
    }
    dataset = np.array([
        [1,0,0],
        [1,0,1],
        [1,1,0],
        [1,1,1]
    ])

 logic_and = compute('logic_and', logic, dataset)
#     logic_or = compute('logic_or', logic, dataset)
#     logic_not = compute('logic_not', logic, [[1,0],[1,1]])
    logic_nand = compute('logic_nand', logic, da   taset)
#     logic_nor = compute('logic_nor', logic, dataset)
    # logic_xor = compute('logic_xor', logic, dataset)
    # logic_xnor = compute('logic_xnor', logic, dataset

 def template(dataset, name, data):
        # act = name[6:]
        print("Logic Function: {}".format(name[6:].upper()))
        print("X0\tX1\tX2\tY")
        toPrint = ["{1}\t{2}\t{3}\t{0}".format(output, *datas) for datas, output in zip(dataset, data)]
        for i in toPrint:
            print(i)
    gates = [logic_and, logic_nand]

    for i in gates:
        template(dataset, *i)

if __name__ == '__main__':
    main()
