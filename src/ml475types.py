from abc import ABCMeta, abstractmethod
from collections import defaultdict
#import numpy as np IF WE USE NUMPY

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self._class_label = label
        
    def __str__(self):
        return str(self._class_label)
    
#Added by me   
class Cluster:
    def __init__(self, instances, mean_vector):
        self._instances = instances
        self._mean_vector = mean_vector

class FeatureVector:
    def __init__(self):
        self._fv = {}
        pass
        
    def add(self, index, value):
        self._fv[index] = value
        pass
        
    def get(self, index):
        return self._fv.get(index, 0.0)
        

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label
        
#Added by me
class MeanVector:
    def __init__(self):
        self._mv = {}
        #self._mv = np.zeros() IF WE USE NUMPY
        pass
        
    def add(self, index, value):
        self._mv[index] = value #OK, BUT CAN PROBABLY OPTIMIZE IF WE USE NUMPY
        pass
        
    def get(self, instances, index):
        return self._mv.get(index, 0.0) 
        #return np.take(instances, index) IF WE USE NUMPY
        
#Added for csvToText in final project
class FeatureMap:
    def __init__(self, featureName):
        self._featureName = featureName
        self._nextInteger = 1
        self._strToInt = defaultdict()
        self._intToStr = defaultdict()
        
    # When we need an integer value associated with a string value we call this method
    # if the string value is not yet mapped to an integer, then we perform that mapping
    # then return the correspond integer
    def getIntegerMapping(self, stringValue):
        integerRepresentation = self._strToInt.get(stringValue)
        if integerRepresentation == None:
            self._strToInt[stringValue] = self._nextInteger
            self._intToStr[self._nextInteger] = stringValue
            self._nextInteger += 1
            return self._strToInt[stringValue]
        else :
            return integerRepresentation
        
    def getStringMapping(self, integerRepresentation):
        return self._intToStr[integerRepresentation]
        

# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances, online_training_iterations, pegasos_lambda, knn): pass 
    
    @abstractmethod
    def predict(self, instance): pass