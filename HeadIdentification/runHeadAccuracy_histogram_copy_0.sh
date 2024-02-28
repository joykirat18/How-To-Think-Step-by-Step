numExamples=100
alternateExamples=50
device="cuda:2"
modelPath=''
python3 IndividualHeadAccuracy_histogram_copy.py --noiseIndex 0 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_copy.py --noiseIndex 1 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_copy.py --noiseIndex 2 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_copy.py --noiseIndex 3 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_copy.py --noiseIndex 4 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_copy.py --noiseIndex 5 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_copy.py --noiseIndex 6 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_copy.py --noiseIndex 7 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_copy.py --noiseIndex 8 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_copy.py --noiseIndex 9 --device $device --numExamples $numExamples --alternateExamples $alternateExamples