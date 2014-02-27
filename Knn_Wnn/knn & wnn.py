#!/usr/bin/python
# -*- coding: utf-8 -*-

from math import sqrt, pow, exp

'''Euclidian distance between two vectors'''
def euclidian_distance(x, y):

    return sqrt(sum([pow(x[i] - y[i],2) for i in range(len(x))]))




'''Euclidian distance between a vector (query) and a set os vectors (dataset)'''
def euclidian_distances(query, dataset):

    return [(euclidian_distance(query, dataset[i]), dataset[i][-1]) for i in range(len(dataset))]




'''K-nearest neighbors algorithm to discrete data'''
def knn_discrete(query, vectors, k=3):

    answer_index = len(vectors[0])-1

    distances = euclidian_distances(query, vectors)
    distances.sort()
    answers = [j for i, j in distances[:k]]
    
    choices = [(answers.count(i), i) for i in set(answers)]

    return max(choices)[1]




'''K-nearest neighbors algorithm to continuous data'''
def knn_continuous(query, vectors, k=3):

    answer_index = len(vectors[0]) - 1

    distances = euclidian_distances(query, vectors)
    distances.sort()
    answers = [answer for distance,answer in distances[:k]]

    return sum(answers) / len(answers)



'''Weighted nearest neighbors algorithm to discrete data'''
def wnn_discrete(query, vectors, k=3):

    answer_index = len(vectors[0]) - 1

    distances = euclidian_distances(query, vectors)
    weighted_distances = []

    for distance,answer in distances:
        try:
            weighted_distances += [(answer / pow(distance, 2),answer)]
        except ZeroDivisionError:
            weighted_distances += [(answer, answer)]

    weighted_distances.sort(reverse=True)
    answers = list(set([answer for weighted_distance, answer in weighted_distances[:k]]))
    
    choices = []
    for possible_final_answer in answers:
        choices += [(sum([weighted_distance for weighted_distance,answer in weighted_distances[:k] if answer == possible_final_answer]), possible_final_answer)]
    
    return max(choices)[1]

    


'''Weighted nearest neighbors algorithm to continuous data'''
def wnn_continuous(query, vectors, k=120, variance=0.1):

    answer_index = len(vectors[0]) - 1

    distances = euclidian_distances(query, vectors)
    weights = [exp(-pow(distance, 2) / (2 * pow(variance, 2))) for distance, answer in distances]
    
    weighted_distances = [(distances[i][1] * weights[i], weights[i]) for i in range(len(distances))]
    weighted_distances.sort(reverse=True)
    
    numerator = sum([weighted_distances[i][0] for i in range(len(weighted_distances[:k]))])
    denominator = sum([weighted_distances[i][1] for i in range(len(weighted_distances[:k]))])

    return numerator / denominator
    
        


#########################################################################

if __name__ == "__main__":
    
    fp = open("iris4knn.dat",'r')
    lines = fp.readlines() 
    fp.close()

    vectors = [line.split() for line in lines]

    for i in range(len(vectors)):
        for j in range(len(vectors[i])):
            vectors[i][j] = float(vectors[i][j])

    print(knn_continuous([5.8, 2.7, 5.1, 1.9], vectors))
