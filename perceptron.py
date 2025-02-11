#Basic Perceptron Algorithm

inputArr = [1, 0, 1, 0, 0]
weights = [2, 3, 1, 4, 5]
output = 0
threshold = 5
weightedSum = 0

for i in range(0, 5):
    
    weightedSum += inputArr[i] * weights[i]
    
if( weightedSum > threshold):
    output = 1
    
print( "output: " , output , " weightedSum: " , weightedSum , "threshold: " , threshold)