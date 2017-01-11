import csv
import random
import numpy as np
import pandas as pd

def Load_Csv(filename):
    f = open(filename)
    dataset = []
    for row in csv.reader(f):
        dataset.append(row)
    del dataset[0]
    for x in range(len(dataset)):
        dataset[x][:] = [float(x) for x in dataset[x][:]]
    for x in range(len(dataset)):
        del dataset[x][0]
        dataset[x][0:30] = [dataset[x][0:30]]
        dataset[x][1:] = [dataset[x][1:]]
    return dataset

def Split_Data(dataset, train_ratio, validation_ratio):
    train_set_size = int(len(dataset)*train_ratio)
    validation_set_size = int(len(dataset)*validation_ratio)
    train_set = []
    validation_set = []
    copy = dataset
    while len(train_set) < train_set_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    while len(validation_set) < validation_set_size:
        index = random.randrange(len(copy))
        validation_set.append(copy.pop(index))
    test_set = copy
    return [train_set, validation_set, test_set]




random.seed(3316)
def Random_Number(a, b):
    return a + (b-a)*random.random()

def Sigmoid(x):
    return 1.0 / (1.0+np.exp(-x))

def Matrix(num_row, num_col, fill=0.0):
    matrix = []
    for i in range(num_row):
        matrix.append([fill]*num_col)
    return matrix




class Artificial_Neural_Networks:

    def __init__(self, ni, nh, no):
        ## node
        ## (+ 1) for bias
        self.ni = ni + 1
        self.nh = nh + 1
        self.no = no
        ## activation
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        ## weight 
        ## wi: between input-layer and hidden-layer 
        ## wo: between hidden-layer and output-layer
        self.wi = Matrix(self.ni, self.nh-1)
        self.wo = Matrix(self.nh, self.no)
        for i in range(self.ni):
            for j in range(self.nh-1):
                self.wi[i][j] = Random_Number(-0.5, 0.5)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = Random_Number(-0.5, 0.5)
        ## change of weight 
        ## for momentum
        self.ci = Matrix(self.ni, self.nh)
        self.co = Matrix(self.nh, self.no)

    def Feed_Forward(self, inputs):
        ## activation of input-layer
        ## self.ai becomes output of input-layer
        for i in range(self.ni-1):
            self.ai[i] = inputs[i]
        ## activation of hidden-layer
        ## self.ah becomes output of hidden-layer
        for j in range(self.nh-1):
            net = 0.0
            for i in range(self.ni):
                net += self.ai[i]*self.wi[i][j]
            self.ah[j] = Sigmoid(net)
        ## activation of output-layer
        ## self.ao becomes output of output-layer
        for k in range(self.no):
            net = 0.0
            for j in range(self.nh):
                net += self.ah[j]*self.wo[j][k]
            self.ao[k] = Sigmoid(net)
        return self.ao

    def Cost_Function(self, targets):
        cost_function = 0.0
        for k in range(len(targets)):
            cost_function += 0.5*(targets[k]-self.ao[k])**2
        return cost_function

    def Back_Propagation(self, targets, learning_rate, momentum):
        ## delta from output-layer
        delta_output = [0.0]*self.no
        for k in range(self.no):
            delta_output[k] = (targets[k]-self.ao[k])*self.ao[k]*(1-self.ao[k])
        ## delta from hidden-layer
        delta_hidden = [0.0]*(self.nh-1)
        for j in range(self.nh-1):
            e = 0.0
            for k in range(self.no):
                e += delta_output[k]*self.wo[j][k]*self.ah[j]*(1-self.ah[j])
            delta_hidden[j] = e
        ## update wo
        for j in range(self.nh):
            for k in range(self.no):
                change = delta_output[k]*self.ah[j]
                self.wo[j][k] += learning_rate*change + momentum*self.co[j][k]
                self.co[j][k] = change
        ## update wi
        for i in range(self.ni):
            for j in range(self.nh-1):
                change = delta_hidden[j]*self.ai[i]
                self.wi[i][j] += learning_rate*change + momentum*self.ci[i][j]
                self.ci[i][j] = change
   
    def Cross_Validation(self, train_set, validation_set, max_epoch, learning_rate, momentum):
        result = []
        for x in range(1, max_epoch+1):
            ## train
            train_cost_function = 0.0
            for data in train_set:
                inputs = data[0]
                targets = data[1]
                self.Feed_Forward(inputs)
                train_cost_function += self.Cost_Function(targets)
                self.Back_Propagation(targets, learning_rate, momentum)
            ## validation
            validation_cost_function = 0.0
            for data in validation_set:
                inputs = data[0]
                targets = data[1]
                self.Feed_Forward(inputs)
                validation_cost_function += self.Cost_Function(targets)
            ## for plotting
            result.append([x, train_cost_function, validation_cost_function])
        result = pd.DataFrame(result)
        result.to_csv("_wdbc_cost_function.csv")

    def Weight(self):
        print("Weight from input layer to hidden layer:")
        for i in range(self.ni):
            print(self.wi[i])
        print("Weight from hidden layer to output layer:")
        for j in range(self.nh):
            print(self.wo[j])

    def Accuracy(self, test_set):
        count = 0
        for data in test_set: 
            inputs = data[0]
            targets = data[1]
            predict = self.Feed_Forward(data[0])

            for x in range(len(predict)):
                if predict[x] == max(predict):
                    predict[x] = 1
                else:
                    predict[x] = 0  
            if targets == predict:
                count += 1
            accuracy = count/float(len(test_set))*100
        print("Accuracy: ", accuracy)


        

def Wdbc():
    ##
    filename = "wdbc_1_of_c_coding.csv"
    dataset = Load_Csv(filename)

    ##
    train_ratio, validation_ratio, test_ratio = [0.4, 0.3, 0.3]
    train_set, validation_set, test_set = Split_Data(dataset, train_ratio, validation_ratio)
    print("Train data set: ", len(train_set), "rows")
    print("Validation data set: ", len(validation_set), "rows")
    print("Test data set: ", len(test_set), "rows")

    

    model = Artificial_Neural_Networks(30,10,2)
            
    ##
    max_epoch, learning_rate, momentum = [1000, 0.0001, 0.8]
    model.Cross_Validation(train_set, validation_set, max_epoch, learning_rate, momentum)
    ##
    #model.Weight()
    ##
    model.Accuracy(test_set)

        
    

if __name__ == "__main__":
    Wdbc()