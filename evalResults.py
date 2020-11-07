import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import os
from hdfs import InsecureClient


client_hdfs = InsecureClient('http://localhost:9870')
dt = ""
rf = ""

with client_hdfs.read('/kth/resultsDT.csv/part-00000-0e86f519-32bb-4ea0-9eb0-631bde673320-c000.csv', encoding = 'utf-8') as reader:
    dt = pd.read_csv(reader)

with client_hdfs.read('/kth/resultsRF.csv/part-00000-691eeb6e-140e-4701-87bb-c94d7cef3bc0-c000.csv', encoding = 'utf-8') as reader:
    rf = pd.read_csv(reader)


resultsDT = dt
resultsRF = rf


#csv_file=dt
#resultsDT = pd.read_csv(csv_file)
resultsDT.columns =['pred','label','V1','V2'] 

#csv_file2=rf
#resultsRF = pd.read_csv(csv_file2)
resultsRF.columns =['pred','label','V1','V2'] 





# iterate through each row in column pred
fraud = 0
notFraud = 0
for row in resultsDT['label']:
    if(row == 0):
        notFraud = notFraud + 1
    else:
        fraud = fraud + 1
xAxis = ["not fraud", "fraud"]
yAxis = [notFraud, fraud]
#print(xAxis, yAxis)
plt.bar(xAxis,yAxis)
plt.xlabel('fraud or not')
plt.ylabel('number of transactions')
plt.title('Data')
# plt.show()

 

#print(resultsDT['label'])

pred = resultsDT['pred']
label = resultsDT['label']

# want to compare them in some way?? not a scatter plot but how
x=list(pred)
y=list(label)
plt.scatter(x,y)
plt.xlabel('pred')
plt.ylabel('label')
plt.title('Data')
# plt.show()



def showErrorRate(data):

    counter = 0
    errors = 0

    for index, row in data.iterrows():
        counter += 1
        if row.pred != row.label:
            errors += 1
        
    errorRate = errors/counter
    return str(errorRate)



def showFalsePositives(data):

    counter = 0
    errors = 0

    for index, row in data.iterrows():
        if row.label == 0:
            counter += 1
        if row.pred == 1 and row.label == 0:
            errors += 1
        
    errorRate = errors/counter
    return str(errorRate)

def showFalseNegatives(data):

    counter = 0
    errors = 0

    for index, row in data.iterrows():
        if row.label == 1:
            counter += 1
        if row.pred == 0 and row.label == 1:
            errors += 1
        
    errorRate = errors/counter
    return str(errorRate)



def showFraud(resultsDTPrint): 
    # These are the detected frauds.
    resultsDTPrint = resultsDT[resultsDT.pred.isin([1])]
    print("Table showing detected frauds")
    print(tabulate(resultsDTPrint, headers=['row','pred','label','V1','V2','V3','V4','V5']))

print("ResultsDT Frauds ") 
showFraud(resultsDT)
print("\n")

print("ResultsRF Frauds ") 
showFraud(resultsRF)
print("ResultsDT errorrate " + showErrorRate(resultsDT))
print("ResultsDT false positive " + showFalsePositives(resultsDT))
print("ResultsDT false negative " + showFalseNegatives(resultsDT))

print("\n")

print("ResultsRF errorrate " + showErrorRate(resultsRF))
print("ResultsRF false positive " + showFalsePositives(resultsRF))
print("ResultsRF false negative " +  showFalseNegatives(resultsRF))



