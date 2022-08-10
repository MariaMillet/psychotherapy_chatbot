import os
path = "data/respGenDataset"
os.chdir(path)
import gdown

id = "1-HLgylzSs2TfRgzYLbhKBe1nRojJcVJF"
gdown.download(id=id)

path = "train"
os.chdir(path)

id = "1-JVZQrBTxt3_OSSeT4npf-HI2rzMUnIt" 
gdown.download(id=id)

id = "1-k_Si6MHqqJNhYJSbneRtvTxP24heD_7"
gdown.download(id=id)
id ="1-SHcbjqYHhQOAVagzRuZUzLLVtLecjy5"
gdown.download(id=id)

os.chdir('../')
print(os.getcwd())
os.chdir('../')
print(os.getcwd())

path = "../data/respSyntheticDataset"
os.chdir(path)
import gdown

# https://drive.google.com/file/d/11ENN2dq2QKEM7SUFS7egQAf0Z8N315kp/view?usp=sharing
# https://drive.google.com/file/d/11HjDBs9GhIHp5aR5SUPO2Kn2J9RGXI1U/view?usp=sharing
# https://drive.google.com/file/d/11F3XiDdawUCaWznLo2XTfdlo1QAZ98jf/view?usp=sharing


id = "11ENN2dq2QKEM7SUFS7egQAf0Z8N315kp"
gdown.download(id=id)


id = "11HjDBs9GhIHp5aR5SUPO2Kn2J9RGXI1U" 
gdown.download(id=id)

id = "11F3XiDdawUCaWznLo2XTfdlo1QAZ98jf"
gdown.download(id=id)
