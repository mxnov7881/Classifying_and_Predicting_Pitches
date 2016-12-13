# read in csv raw data files and convert to txt file
# there are several dictionaries used to map string values to integer values
#
# The extra info list is used

import csv
from ml475types import FeatureMap
from collections import defaultdict
from collections import deque
with open('LesterTest.csv') as csvfile:
    extraInfo = []
    ignoreFile = open('PredictionIgnore.txt', 'r')
    mappedFeatures = open('mapCols.txt', 'r')
    # for every line in the txt file, add that string to the extraInfo list
    for line in ignoreFile:
        line = line.replace("\n", "")
        extraInfo.append(line)

    maps = defaultdict() #mapping of column names to dictionaries containing their mappings
    
    # create a dictionary for every column that requires a mapping
    for line in mappedFeatures:
        line = line.replace("\n", "")
        maps[line] = FeatureMap(line)

    

    pitchKeyStrToInt = defaultdict() #mapping of pitch ml475types to integers
    pitchKeyIntToStr = defaultdict()
    numPitchTypes = 1
    reader = csv.DictReader(csvfile)
    
    for row in reader:
        if row['mlbam_pitch_name'] not in pitchKeyStrToInt:
            pitchKeyStrToInt[row['mlbam_pitch_name']] = numPitchTypes
            pitchKeyIntToStr[numPitchTypes] = row['mlbam_pitch_name']
            numPitchTypes = numPitchTypes + 1
    
    for intkey in pitchKeyIntToStr:
        outputStr = "LesterPrediction" + pitchKeyIntToStr[intkey] + ".test"
        tf = open(outputStr, 'w')
        with open('LesterTest.csv') as csvfile:
            
            # use a list as a queue to store the pitches thrown in the AB so far
            # append to the end of the feature vector 
            pitchQueue = deque()
            
            reader = csv.DictReader(csvfile)
            for row in reader:
                #Maintain a dictionary mapping pitch ml475types to integers starting at 0
                #These are our instance labels
                #Start each line by writing the label value
                
                # if curAtBat = 1 then it is a new batter, clear the pitchQueue
                curAtBat = row['ab_count']
                if curAtBat == str(1):
                    pitchQueue = deque()
                
                if pitchKeyStrToInt[row['mlbam_pitch_name']] == intkey:
                    tf.write(str(1) + ' ')
                else:
                    tf.write(str(0) + ' ')
                pitchInt = pitchKeyStrToInt[row['mlbam_pitch_name']]
                #Remove the key from the row dictionary to avoid repeating it in the txt file
                del row['mlbam_pitch_name']
                #Now print the feature values prefixed by their corresponding index in the fv
                index = 1
                for data in row:
                    # ignore features that are in the ignore.txt file
                    if data not in extraInfo:
                        featMap = maps.get(data)
                        if featMap == None:
                            tf.write(str(index) + ':' + str(row[data]) + ' ')
                            index += 1
                        else:
                            tf.write(str(index) + ':' + str(featMap.getIntegerMapping(row[data])) + ' ')
                            index += 1
                for pitchInteger in pitchQueue:
                    tf.write(str(index) + ':' + str(pitchInteger) + ' ')
                    index += 1
                pitchQueue.append(pitchInt)
                tf.write('\n')
        tf.close()
print('done')

# TODO
# - print the index before each ValueError
# - create dictionaries mapping string vals to integers
#    --> is this proper? seems like there would be a problem
#        because the scale of these mappings will directly
#        affect the distance calculations being made?
#            ...maybe not, could try to prove or disprove mathematically
#                (think orthogonality),
#                or test directly if classifications change when changing
#                the scaling of these mappings.
#        *
#        |    *    *        *
#        |        *        *                *
#        |
#--------+---------
#        |
#        |
#        |
#        |
