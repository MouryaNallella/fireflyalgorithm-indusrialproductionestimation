import numpy as np
from flask import Flask, request, render_template
import pickle
import os
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import xlsxwriter
class FireflyAlgorithm():

    def __init__(self, D, NP, nFES, alpha, betamin, gamma, LB, UB, function,data_set):
        self.D = D  # dimension of the problem
        self.NP = NP  # population size
        self.nFES = nFES  # number of function evaluations
        self.alpha = alpha  # alpha parameter
        self.betamin = betamin  # beta parameter
        self.gamma = gamma  # gamma parameter
        self.data_set=data_set
        # sort of fireflies according to fitness value
        self.Index = [0] * self.NP
        self.Fireflies = self.data_set.copy()# firefly agents
        self.Fireflies_tmp =self.data_set.copy()  # intermediate pop
        self.Fitness = [0.0] * self.NP  # fitness values
        self.I = [0.0] * self.NP  # light intensity
        self.nbest = [0.0] * self.NP  # the best solution found so far
        self.LB = LB  # lower bound
        self.UB = UB  # upper bound
        self.fbest = None  # the best
        self.evaluations = 0
        self.Fun = function
        self.time_gain=0


    def init_ffa(self):
        for i in range(self.NP):
            for j in range(self.D):
                self.Fireflies[i][j] = random.uniform(
                    20, 30) +self.Fireflies[i][j] # adding noise of 20-30 for better movement and hence better Solution
            self.Fitness[i]=random.uniform(30,60)

            self.I[i] = self.Fitness[i]

    def alpha_new(self, a):
        delta = 1.0 - math.pow((math.pow(10.0, -4.0) / 0.9), 1.0 / float(a))
        return (1 - delta) * self.alpha

    def sort_ffa(self):  # implementation of bubble sort
        for i in range(self.NP):
            self.Index[i] = i

        for i in range(0, (self.NP - 1)):
            j = i + 1
            for j in range(j, self.NP):
                if (self.I[i] > self.I[j]):
                    z = self.I[i]  # exchange attractiveness
                    self.I[i] = self.I[j]
                    self.I[j] = z
                    z = self.Fitness[i]  # exchange fitness
                    self.Fitness[i] = self.Fitness[j]
                    self.Fitness[j] = z
                    z = self.Index[i]  # exchange indexes
                    self.Index[i] = self.Index[j]
                    self.Index[j] = z


    def replace_ffa(self):  # replace the old population according to the new Index values
        # copy original population to a temporary area so that when we chnage data of current firefly Array the old value preserved to calculate the delta for other fireflies
        for i in range(self.NP):
            for j in range(self.D):
                self.Fireflies_tmp[i][j] = self.Fireflies[i][j]

        # generational selection in the sense of an EA
        for i in range(self.NP):
            for j in range(self.D):
                self.Fireflies[i][j] = self.Fireflies_tmp[self.Index[i]][j]

    def FindLimits(self, k):
        for i in range(self.D):
            if self.Fireflies[k][i] < self.LB:
                self.Fireflies[k][i] = self.LB
            if self.Fireflies[k][i] > self.UB:
                self.Fireflies[k][i] = self.UB

    def move_ffa(self):
        for i in range(self.NP):
            scale = abs(self.UB - self.LB)
            for j in range(self.NP):
                r = 0.0
                for k in range(self.D):
                    r += (self.Fireflies[i][k] - self.Fireflies[j][k]) * \
                        (self.Fireflies[i][k] - self.Fireflies[j][k])
                r = math.sqrt(r)
                if self.I[i] > self.I[j]:  # brighter and more attractive
                    beta0 = 1.0
                    beta = (beta0 - self.betamin) * \
                        math.exp(-self.gamma * math.pow(r, 2.0)) + self.betamin # b=b0* e^(-yr^2)
                    for k in range(self.D):
                        r = random.uniform(0, 1)
                        tmpf = self.alpha * (r - 0.5) * scale
                        self.Fireflies[i][k] = self.Fireflies[i][
                            k] * (1.0 - beta) + self.Fireflies_tmp[j][k] * beta + tmpf
            self.FindLimits(i)

    def Run(self):
        self.init_ffa()

        while self.evaluations < self.nFES:

            # optional reducing of alpha
            self.alpha = self.alpha_new(self.nFES/self.NP)

            # evaluate new solutions
            for i in range(self.NP):
                self.Fitness[i] = self.Fun(self.D, self.Fireflies[i])
                self.evaluations = self.evaluations + 1
                self.I[i] = self.Fitness[i]

            # ranking fireflies by their light intensity
            self.sort_ffa()
            # replace old population
            self.replace_ffa()
            # find the current best
            self.fbest = self.I[0]
            self.bestI=np.argmax(self.I)
            self.nbest=self.Fireflies[self.bestI]

            # move all fireflies to the better locations
            self.move_ffa()
        return  self.nbest, self.fbest
df=np.random.randint(30,100,(15,3)) # 20creating a Random dataset for 15 machines
df=pd.DataFrame(df,columns =['rate','sales_in_first_month','sales_in_second_month'])
df.to_csv('sales.csv')
df=df.values
prob_dim=(len(df))
pop_size=3000
data_set=[[0 for x in range(prob_dim)] for x in range(pop_size)]

def generate(df):
    no_of_lane=prob_dim
    for i in range(pop_size):
        for j in range(no_of_lane):
            data_set[i][j]=((df[j][0]*df[j][1])//df[j][2])#using random created dataset for initializing the fireflies 
            # print(data_set[i][j])
    return data_set


data_set=generate(df)
# print(data_set)

def function(D, sol): #a fitness function for fireflies 
    val = 0.0
    for i in range(D):
        val = val + sol[i] * sol[i]
    return val

Algorithm = FireflyAlgorithm(10, 10, 100, 0.5, 0.2,0.9,30, 180, function,data_set)

Best = Algorithm.Run()
data = pd.DataFrame({'Values': Best[0]})
datatoexcel = pd.ExcelWriter("excel.xlsx",engine='xlsxwriter')
data.to_excel(datatoexcel,sheet_name='Sheet1')
datatoexcel.save()
plt.figure(figsize=(5,5)) #increasing figsize for better visualization
plt.bar([x+1 for x in range(prob_dim)],Algorithm.nbest)#1st list is the traffic light no
plt.title("Graph of Machines and their alocated timing")
plt.xlabel('Machines Used')
plt.ylabel('allocated time')
plt.savefig("C:/Users/Store/Desktop/mini/static/pics/graph.jpg")
print ('best solution is ',Best[0],'and best fiteness ',Best[1])

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
picFolder = os.path.join('static','pics')
app.config['UPLOAD_FOLDER'] = picFolder

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/algorithm',methods=['POST'])
def algorithm():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    graph =os.path.join(app.config['UPLOAD_FOLDER'],'graph.jpg')
    return render_template('Graph.html', user_image=graph, prediction_text='Next Month Production should be  {}'.format(output))
if __name__ == "__main__":
    app.run(debug=True)