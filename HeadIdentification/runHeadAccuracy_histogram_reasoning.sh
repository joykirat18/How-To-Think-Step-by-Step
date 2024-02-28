numExamples=100
alternateExamples=50
device="cuda:1"
modelPath='/home/models/vicuna-7b'
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 0 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 1 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 2 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 3 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 4 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 5 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 6 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 7 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 8 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 9 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 10 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 11 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 12 --device $device --numExamples $numExamples --alternateExamples $alternateExamples
python3 IndividualHeadAccuracy_histogram_reason.py --noiseIndex 13 --device $device --numExamples $numExamples --alternateExamples $alternateExamples