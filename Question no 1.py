
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt






def CostOfTheClassification (WeightagesOftheFeatures, FeaturesOftheData, LabelsOftheData):

    WeightagesOftheFeatures = np.matrix(WeightagesOftheFeatures)
    
    FeaturesOftheData = np.matrix(FeaturesOftheData)
    
    LabelsOftheData = np.matrix(LabelsOftheData)
    


    number = np.dot(FeaturesOftheData, WeightagesOftheFeatures.T)
    
    ExpectedValueFromtheSigmoidFunction = SigmoidFunctionToGetProbabilityOftheStudentTogetAddmission(number)
    

    numberVartoCalculate = (np.multiply(LabelsOftheData, np.log(ExpectedValueFromtheSigmoidFunction))) + (np.multiply((1 - LabelsOftheData), np.log(1 - ExpectedValueFromtheSigmoidFunction)))
    
    ResultOfCost = -(np.sum(numberVartoCalculate))/len(FeaturesOftheData)

    return ResultOfCost








def ResultsOftheClassificationByLogisticRegression(OptimalTheetasArrayforClassification, TestCaseWithfeaturesArrayX):

    ProbabilityOftheStudentTogetAddmission = SigmoidFunctionToGetProbabilityOftheStudentTogetAddmission(np.dot(TestCaseWithfeaturesArrayX, OptimalTheetasArrayforClassification.T))
    

    if ProbabilityOftheStudentTogetAddmission >=0.5:
        
        print ('Probability of getting admission'+ str(ProbabilityOftheStudentTogetAddmission) )
        
        
        
    else:
        
        print('Probability not getting admission' + str(ProbabilityOftheStudentTogetAddmission))
        
        

def SigmoidFunctionToGetProbabilityOftheStudentTogetAddmission(x):

    ValueGivenbySigmoidFunction = 1 / (1 + np.exp(-x))
    
    
    return ValueGivenbySigmoidFunction





def LogisticgradientDescent (WeightagesOftheFeatures, FeaturesOftheData, LabelsOftheData):

    WeightagesOftheFeatures = np.matrix(WeightagesOftheFeatures)
    
    FeaturesOftheData = np.matrix(FeaturesOftheData)
    
    LabelsOftheData = np.matrix(LabelsOftheData)
    

    FeaturesArray = FeaturesOftheData.shape[1]

    lengthOfFeaturesArray = len(FeaturesOftheData)


    EquationResultAfterMultiplyingTheetasWithFeatures = np.dot(FeaturesOftheData, WeightagesOftheFeatures.T)
    
    
    
    ExpectedValueFromtheSigmoidFunction = SigmoidFunctionToGetProbabilityOftheStudentTogetAddmission(EquationResultAfterMultiplyingTheetasWithFeatures)

    CostAfterSubtractionFromOriginalLabel = np.subtract(ExpectedValueFromtheSigmoidFunction, LabelsOftheData)
    
    
    DescentAddedToFeaturesArray = np.zeros(FeaturesArray)
    
    

    for i in range (FeaturesArray):
        
        
        FinalResultForTheEquationObtainedAfterMultiplcation = np.multiply(CostAfterSubtractionFromOriginalLabel, FeaturesOftheData[:, i])
        
        
        DescentAddedToFeaturesArray[i] = (np.sum(FinalResultForTheEquationObtainedAfterMultiplcation)/lengthOfFeaturesArray)
        
        


    return DescentAddedToFeaturesArray




df = pd.read_csv('ex2data1.txt', names=['Exam1', 'Exam2', 'Classes'])

FeaturesExtrcatedFromFile = df.as_matrix(columns = ['Exam1','Exam2'])      


LabelsOftheDataExtractedFromFile = df.as_matrix(columns = ['Classes'])             



LabelsThatShowStudentIsAddmitted = np.where(LabelsOftheDataExtractedFromFile==1)[0]


LabelsThatShowStudentIsNotAddmitted = np.where(LabelsOftheDataExtractedFromFile==0)[0]



plt.scatter(FeaturesExtrcatedFromFile[LabelsThatShowStudentIsAddmitted, 0], FeaturesExtrcatedFromFile[LabelsThatShowStudentIsAddmitted, 1], c='black', alpha=1, marker='+', s=45, label='Admitted')

plt.scatter(FeaturesExtrcatedFromFile[LabelsThatShowStudentIsNotAddmitted, 0], FeaturesExtrcatedFromFile[LabelsThatShowStudentIsNotAddmitted, 1], c='yellow', alpha=1, marker='o', s=45, label='Not Admitted')

plt.xlabel('Exam 1 score')

plt.ylabel('Exam 2 score')

plt.legend(['Admitted','Not Admitted'])


plt.axis([30, 100, 30, 100])

plt.show()


sizeofFeaturesfromFile = len(FeaturesExtrcatedFromFile)


FeaturesArrayures = len(FeaturesExtrcatedFromFile[1,:]) + 1


FeaturesExtrcatedFromFile = np.append(np.ones((FeaturesExtrcatedFromFile.shape[0],1)),FeaturesExtrcatedFromFile,axis=1)     


Theta = np.zeros(FeaturesArrayures)                      




result = opt.fmin_tnc(func=CostOfTheClassification, x0=Theta, fprime=LogisticgradientDescent, args=(FeaturesExtrcatedFromFile,LabelsOftheDataExtractedFromFile))

OptTheta = np.matrix(result[0])






optiCost = CostOfTheClassification(OptTheta, FeaturesExtrcatedFromFile, LabelsOftheDataExtractedFromFile)




test=np.matrix([1,45,85]) 

ResultsOftheClassificationByLogisticRegression(OptTheta, test)




df = pd.read_csv('ex2data1.txt', names=['Exam1', 'Exam2', 'Classes'])

FeaturesExtrcatedFromFile = df.as_matrix(columns = ['Exam1','Exam2'])     



LabelsOftheDataExtractedFromFile = df.as_matrix(columns = ['Classes'])            



LabelsThatShowStudentIsAddmitted = np.where(LabelsOftheDataExtractedFromFile==1)[0]


LabelsThatShowStudentIsNotAddmitted = np.where(LabelsOftheDataExtractedFromFile==0)[0]



plt.scatter(FeaturesExtrcatedFromFile[LabelsThatShowStudentIsAddmitted, 0], FeaturesExtrcatedFromFile[LabelsThatShowStudentIsAddmitted, 1], c='black', alpha=1, marker='+', s=45, label='Admitted')


plt.scatter(FeaturesExtrcatedFromFile[LabelsThatShowStudentIsNotAddmitted, 0], FeaturesExtrcatedFromFile[LabelsThatShowStudentIsNotAddmitted, 1], c='yellow', alpha=1, marker='o', s=45, label='Not Admitted')


plt.xlabel('E1')

plt.ylabel('E2')

plt.legend(['Admitted','Not Admitted'])


plt.axis([30, 100, 30, 100])





point1=(-1.0*OptTheta[0,0])/OptTheta[0,2]      


point2=(-1.0*OptTheta[0,0])/OptTheta[0,1]       

plotX=np.array([0.0,point1])


plotY=np.array([point2,0.0])


plt.plot(plotX,plotY,label='Decision Boundry')

plt.show()



print('After The optimization process Cost is = ' + str(round(optiCost, 3)))
print('After The optimization process Thetas are as these' +' %s' % OptTheta )
