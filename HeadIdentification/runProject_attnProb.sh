numExamples=80
device="cuda:0"
modelPath='/home/models/Llama-2-7b-hf'
python3 project_attnProb.py --noiseIndex 0 --device $device --numExamples $numExamples --modelPath $modelPath
python3 project_attnProb.py --noiseIndex 1 --device $device --numExamples $numExamples --modelPath $modelPath
python3 project_attnProb.py --noiseIndex 2 --device $device --numExamples $numExamples --modelPath $modelPath
python3 project_attnProb.py --noiseIndex 3 --device $device --numExamples $numExamples --modelPath $modelPath
python3 project_attnProb.py --noiseIndex 4 --device $device --numExamples $numExamples --modelPath $modelPath
python3 project_attnProb.py --noiseIndex 5 --device $device --numExamples $numExamples --modelPath $modelPath
python3 project_attnProb.py --noiseIndex 6 --device $device --numExamples $numExamples --modelPath $modelPath
python3 project_attnProb.py --noiseIndex 7 --device $device --numExamples $numExamples --modelPath $modelPath
python3 project_attnProb.py --noiseIndex 8 --device $device --numExamples $numExamples --modelPath $modelPath
python3 project_attnProb.py --noiseIndex 9 --device $device --numExamples $numExamples --modelPath $modelPath
