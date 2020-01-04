from mylib.models.misc import set_gpu_usage

set_gpu_usage()

import csv
from mylib.dataloader.dataset import ClfSegDataset, get_balanced_loader, get_loader, Transform
from mylib.models import densesharp, metrics, losses

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import numpy as np
from keras.models import load_model, Model
import pandas as pd

def main():
    '''

    :param batch_sizes: the number of examples of each class in a single batch
    :param crop_size: the input size
    :param random_move: the random move in data augmentation
    :param learning_rate: learning rate of the optimizer
    :param segmentation_task_ratio: the weight of segmentation loss in total loss
    :param weight_decay: l2 weight decay
    :param save_folder: where to save the snapshots, tensorflow logs, etc.
    :param epochs: how many epochs to run
    :return:
    '''
    # print(test_data['voxel'].Transform(candi['voxel'],32))
    # print(test_data['seg'].Transform(candi['seg'],32))
    # print(test_data['voxel'].resize(32,32,32))
    # print(test_data['seg'].resize(32,32,32))

    test_dataset = ClfSegDataset(crop_size=[32, 32, 32], subset=[0, 1, 2, 3], move=3,
                                 define_label=lambda l: l)
    test_model: Model = load_model('tmp/test/weights.01.h5',
                                   custom_objects={'dice_loss_100': losses.DiceLoss(),
                                                   'precision': metrics.precision,
                                                   'recall': metrics.recall,
                                                   'fmeasure': metrics.fmeasure,
                                                   'invasion_acc': metrics.invasion_acc,
                                                   'invasion_fmeasure': metrics.invasion_fmeasure,
                                                   'invasion_precision': metrics.invasion_precision,
                                                   'invasion_recall': metrics.invasion_recall,
                                                   'ia_acc': metrics.ia_acc,
                                                   'ia_fmeasure': metrics.ia_fmeasure,
                                                   'ia_precision': metrics.ia_precision,
                                                   'ia_recall': metrics.ia_recall})
    n=0
    list=[]
    for idx in test_dataset.index:
        preds = test_model.predict(np.array([test_dataset[idx][0]]))
        a,b = preds[0][0]
        list.append(b)
        n=n+1
        #print(preds[0][0])
   # tmp=pd.DataFrame(list,columns=['Predicted'])
  #  tmp.to_csv('submission.csv',index=False)
    with open('submission.csv','r') as f:
        reader=csv.reader(f)
        column1 = [row[0] for row in reader]
        f.close()
    with open("submission.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Id','Predicted'])
        for k in range(len(column1)):
            if k!=0:
                writer.writerow([column1[k],list[k-1]])
    csvfile.close()
       # print(preds[0][0][1])
if __name__ == '__main__':
    main()
