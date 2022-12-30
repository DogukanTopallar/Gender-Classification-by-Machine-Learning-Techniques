import math
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from numpy import random
from random import randint
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

data = pd.read_csv(r"C:\Users\Dogukan\Desktop\DERSLER\BM SON SINIF\YAPAY ZEKA\FinalOdev\Gender-Classification-by-Machine-Learning-Tecniques\voice.csv")
columns = data.columns.tolist()
data.label = [1 if each == "female" else 0 for each in data.label]

x_data = data.drop(["label"],axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data)).values
y = data.label.values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)


def KNN():
    knn = KNeighborsClassifier(n_neighbors=2)

    #test-train verileri için x ve y'ler
    x_data = data.drop(["label"],axis=1)
    x = (x_data - np.min(x_data)) / (np.max(x_data)).values
    y = data.label.values

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
    #test_size=0.2 means %20 test datas, %80 train datas
    knn.fit(x_train,y_train)
    
    print("Score for Number of Neighbors = 2: {}".format(knn.score(x_test,y_test)))
    # score = knn.score(x_test,y_test)

    #Confusion Matrix
    y_pred = knn.predict(x_test)
    conf_mat = confusion_matrix(y_test,y_pred)

    #Visualization Confusion Matrix
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("Predicted Values")
    plt.ylabel("True Values")
    plt.show()

KNN()

def KNN_with_HC():
    min = 0
    max = 0
    knn = KNeighborsClassifier(n_neighbors=2)
    for i in range(0,100):
        random_selected_column = random.choice(columns, randint(1 , 20))
        x_data = data.drop(["label"],axis=1)
        x = (x_data - np.min(x_data)) / (np.max(x_data)).values
        x = data[random_selected_column].values
        y = data.label.values
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
        if(len(x_train) != 0 and len(y_train) !=0):
            knn.fit(x_train,y_train)
            y_pred = knn.predict(x_test)
            score = knn.score(x_test,y_test)
            if(i == 0):
                min = score
                max = score
            if(score>max):
                max = score
            if(score<min):
                min=score
            

    print("Min Score With HC for Number of Neighbors = 2: %.6f" % (min*100))
    print("Max Score With HC for Number of Neighbors = 2: %.6f" % (max*100))

# KNN_with_HC()

# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state= 42)
# accuracies = cross_val_score(estimator=knn,X = x_train, y = y_train.ravel(), cv = 5)
# acc = np.mean(accuracies)
# print("average accuracy: ",np.mean(accuracies))
# print("average std: ",np.std(accuracies))

# #%% neigbor parameter has randomly selected 1's
# def get_neighbors(a):
#     Z = np.random.random(54) > 0.5
#     Z = Z.astype(int).astype(str)
    
#     a.columns = Z
#     a = a.xs('1', axis=1)  
#     return a   

# def get_cost(x,y):
#     x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42)
#     knn = KNeighborsClassifier(n_neighbors=2)
#     accuracies = cross_val_score(estimator=knn,X = x_train, y = y_train.ravel(), cv = 5)
#     return np.mean(accuracies)


# def simulated_annealing(param, initial_state, y):
#     """Peforms simulated annealing to find a solution"""
     
#     initial_temp = 4
#     final_temp = .1
#     alpha = 0.01
    
#     current_temp = initial_temp

#     # Start by initializing the current state with the initial state
#     #initial_state is knn score of df with only 1's selected
#     current_state = initial_state    
#     solution = current_state

#     while current_temp >=  final_temp:
#         neighbor = get_neighbors(param)
#         # Check if neighbor is best so far
#         cost_diff = get_cost(neighbor,y)

#         if cost_diff < math.exp(cost_diff / current_temp):
#             solution = cost_diff
#         # decrement the temperature
#         current_temp -= alpha
#     return solution

# #%% Print the score with simulated annealing
# print("score", simulated_annealing(x, acc, y)
def KNN_With_SA():
    def objective_function(solution):
        knn = KNeighborsClassifier(n_neighbors=2)
        if(sum(solution)==0):
            return 0
        knn.fit(x_train,y_train)
        score = knn.score(x_test,y_test)
        return score
        


    def neighborhood_function(solution, obj_val):
        neighbors = []
        for i in range(len(solution)):
            temp_sol = solution.copy()
            temp_sol[i] = ~temp_sol[i]
            if ( objective_function(temp_sol) < obj_val ):
                neighbors.append(temp_sol)
        if len(neighbors) == 0:
            return None
        rand_ind = np.random.randint(0, len(neighbors))
        return neighbors[rand_ind]
        


    initial_temp = 1
    cooling_coef = 0.8

    solution = np.random.rand(600)>0.5
    obj_val = objective_function(solution)

    best_solution = solution.copy()
    best_val = obj_val

    convergence = []
    for i in range(1000):
        candidate_solution = neighborhood_function(solution, obj_val)
        if candidate_solution is None:
            break
        cand_val = objective_function(candidate_solution)
        
        if(cand_val <obj_val ):
            obj_val, solution = cand_val, candidate_solution.copy()
            if(cand_val < best_val):
                best_val, best_solution = cand_val, candidate_solution.copy()
        else:
            if initial_temp/ ( 1 + np.log(i+1) ) > np.random.random():
                obj_val, solution = cand_val, candidate_solution.copy()
        convergence.append(best_val)
        
        
        
        
    print(best_val)

    from matplotlib.pyplot import plot

    plot(convergence)