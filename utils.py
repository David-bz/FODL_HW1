import pandas as pd
import pickle

def create_df_from_lst():
    train_acc_titles = ['train_acc_epoch_{}'.format(i) for i in range(40, -1, -3)]
    train_loss_titles = ['train_loss_epoch_{}'.format(i) for i in range(40, -1, -3)]
    test_acc_titles = ['test_acc_epoch_{}'.format(i) for i in range(40, -1, -3)]
    test_loss_titles = ['test_loss_epoch_{}'.format(i) for i in range(40, -1, -3)]

    titles = ['lr', 'momentum', 'std'] + train_acc_titles + train_loss_titles + test_acc_titles + test_loss_titles + ['training_time']
    with open('./grid_res_2.pkl', "rb") as f:
        l1 = pickle.load(f)
    with open('./grid_res_3.pkl', "rb") as f:
        l2 = pickle.load(f)
    l1 += l2
    df =  pd.DataFrame(l1, columns=titles)
    return df

def create_df_from_lst_avg():

    titles = ['lr', 'momentum', 'std' , 'train_acc_avg' , 'train_loss_avg' , 'test_acc_avg' , 'test_loss_avg', 'training_time']
    with open('./grid_res_4.pkl', "rb") as f:
        l1 = pickle.load(f)
    df =  pd.DataFrame(l1, columns=titles)
    return df