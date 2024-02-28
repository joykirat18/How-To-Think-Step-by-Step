numExamples=100
alternateExamples=40
device="cuda:1"
modelPath='/home/models/vicuna-7b'
# python3 IndividualHeadAccuracy_histogram.py --noiseIndex 0 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram.py --noiseIndex 1 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram.py --noiseIndex 2 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram.py --noiseIndex 3 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram.py --noiseIndex 4 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram.py --noiseIndex 5 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram.py --noiseIndex 6 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram.py --noiseIndex 7 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram.py --noiseIndex 8 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram.py --noiseIndex 9 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
