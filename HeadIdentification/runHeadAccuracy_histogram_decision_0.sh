numExamples=100
alternateExamples=50
device="cuda:0"
modelPath='/home/models/vicuna-7b'
# python3 IndividualHeadAccuracy_histogram_decision.py --noiseIndex 0 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_decision.py --noiseIndex 1 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_decision.py --noiseIndex 2 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_decision.py --noiseIndex 3 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_decision.py --noiseIndex 4 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_decision.py --noiseIndex 5 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_decision.py --noiseIndex 6 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_decision.py --noiseIndex 7 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_decision.py --noiseIndex 8 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
# python3 IndividualHeadAccuracy_histogram_decision.py --noiseIndex 9 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
