import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from numpy import random
from random import randint
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



data = pd.read_csv(r"C:\Users\Dogukan\Desktop\DERSLER\BM SON SINIF\YAPAY ZEKA\FinalOdev\Gender-Classification-by-Machine-Learning-Tecniques\voice.csv")
columns = data.columns.tolist()
data.label = [1 if each == "female" else 0 for each in data.label]


x_data = data.drop(["label"],axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data)).values
y = data.label.values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)

def DecisionTree():
    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(x_train,y_train)
    print("Decision Tree Classification Score: ",dec_tree.score(x_test,y_test))
    # method_names.append("Decision Tree")
    # method_scores.append(dec_tree.score(x_test,y_test))

    #Confusion Matrix
    y_pred = dec_tree.predict(x_test)
    conf_mat = confusion_matrix(y_test,y_pred)
    #Visualization Confusion Matrix
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(conf_mat,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
    plt.xlabel("Predicted Values")
    plt.ylabel("True Values")
    plt.show()

def DecisionTree_With_HC():
    min = 0
    max = 0
    dec_tree = DecisionTreeClassifier()
    for i in range(0,100):
        random_selected_column = random.choice(columns, randint(1 , 20))
        x_data = data.drop(["label"],axis=1)
        x = (x_data - np.min(x_data)) / (np.max(x_data)).values
        x = data[random_selected_column].values
        y = data.label.values
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state = 42)
        if(len(x_train) != 0 and len(y_train) !=0):
            dec_tree.fit(x_train,y_train)
            y_pred = dec_tree.predict(x_test)
            score = dec_tree.score(x_test, y_test)
            #score = accuracy_score(y_test ,y_pred)
            if(i == 0):
                min = score
                max = score
            if(score>max):
                max = score
            if(score<min):
                min=score
    print("Min Decision Tree With HC Classification Score: %.6f " %(min*100))
    print("Max Decision Tree With HC Classification Score: %.6f " %(max*100))


# DecisionTree_With_HC()

def DecTree_With_SA():
    def objective_function(solution):
        dec_tree = DecisionTreeClassifier()
        if(sum(solution)==0):
            return 0
        dec_tree.fit(x_train,y_train)
        score = dec_tree.score(x_test, y_test)
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

DecTree_With_SA()