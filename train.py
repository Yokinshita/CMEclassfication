import model.train_schedule as train_schedule
import model.model_defination as model_defination
import argparse


if __name__ == '__main__':
    # 基本训练参数
    parser = argparse.ArgumentParser(description='CME network training initalizer')
    parser.add_argument('--save_location',help='path to find dataset',type=str,default=r'D:\Programming\CME_data')
    parser.add_argument('-l','--learning_rate',help='learning rate',type=float,default=1e-3)
    parser.add_argument('-b','--batch_size',help='batch size',type=int,default=300)
    parser.add_argument('-e','--num_epochs',help='epochs',type=int,default=50)
    parser.add_argument('--drop_prob',help='dropout probability',type=float,default=0.5)
    parser.add_argument('--step_size',help='period of learning rate decay',type=int,default=1)
    parser.add_argument('--gamma',help='factor of learning rate decay',type=float,default=0.5)
    parser.add_argument('--log_path',help='path to save log',type=str,default=r'.\log')
    parser.add_argument('-a','--aloud',help='whether to output more detailed infomation in terminal while training',action='store_true',default=False)
    args = parser.parse_args()
    # 其他参数
    args.selected_remarks = ['Halo', 'No Remark', 'Partial Halo']
    args.train_percentage = 0.7

    paras = vars(args)
    net = model_defination.vgg19(drop_prob=paras.drop_prob)
    modeltrain = train_schedule.ModelTrain(paras, net)
    modeltrain.fit()
    modeltrain.save_info()
