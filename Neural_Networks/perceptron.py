#!/usr/bin/python
# -*- coding: utf-8 -*-

def f(net,threshold=0.5):
    if(net < threshold):
        return 0
    else:
        return 1
    
def perceptron(dataset, epsilon=0.1, eta=0.1):
    n_attributes = len(dataset[0]) - 1
    answer_index = len(dataset[0]) - 1

    # Generating random weights
    import random
    weights = [random.random() for i in range(n_attributes)]
    bias = random.random()

    error = epsilon + 1.0
    while(error > epsilon):
        error = 0.0
        # Training using each sample
        for sample in dataset:
            #calculating net    
            net = sum([sample[i]*weights[i] for i in range(len(sample)-1)])+bias

            # Calculating the error of a sample
            output = f(net)
            expectedOutput = sample[answer_index]

            error += (expectedOutput - output) ** 2

            # Updating the weights:   weight = weight + step*error*input
            
            for weight in range(len(weights)):
                weights[weight] += 2 * eta * (expectedOutput - net) * sample[weight]
            
            # Updating the bias
            bias += 2 * eta * (expectedOutput - net)

            print str(sample + [output])
        print "error: " + str(error)
    print "weights: " + str(weights) + ", bias: " + str(bias)






###############################################

if __name__ == "__main__":
    
    dataset = [[0,0,0],[0,1,1],[1,0,1],[1,1,1]]
    perceptron(dataset)    

