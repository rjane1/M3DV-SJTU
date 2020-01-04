mylib文件夹、test.py和train.py包含全部代码。

#train.py
模型训练代码，需要调用训练集数据。

在运行代码的时候，需要将mylib/dataloader中的ENVIRON进行修改，将路径变为存放训练集数据的路径。例如：“demo/”，demo文件夹存放于与train.py同一目录下，demo中包含一个info.csv文件记录了name和lable两列数据，一个nodule文件夹内有不同name对应的3D扫描数据信息。



#test.py
模型测试（predicted）代码，需要调用已训练出的模型和测试集数据。

在运行代码时，同样需要将mylib/dataloader中的ENVIRON进行修改，将路径变为存放测试集数据的路径，路径中同样包含文件info.csv和文件夹nodule，其中info.csv中的lable一项全置零即可。

调用模型：模型路径在test.py的load_model函数中，直接修改即可。

输出结果在submission中，将submission.csv放在与test.py同一目录下即可实现改写。



#tmp
tmp/test/weights.01.h5为本项目训练得到的模型，在测试集上准确率约为0.62023。

