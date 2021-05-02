import torch
import argparse,os,sys
import time
import numpy as np
import random
from utils import *
from Model.baseNet import BaseNet

def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="choose the mode",
        epilog="python run.py mode")

    # required input parameters
    parser.add_argument(
        'mode', type=str,
        help='train test train_offkd')
    
    #TRAIN SECTION
    parser.add_argument(
        '--seed', type=int, default=1,
        help='fix random seed')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='set mini-batch size')
    parser.add_argument(
        '--model', type=str, default='lenet5',
        help='choose NeuralNetwork type')
    parser.add_argument(
        '--device', type=str, default='gpu',
        help='choose NeuralNetwork')
    parser.add_argument(
        '--colab', type=bool, default=False,
        help='if you are in colab use it')
    parser.add_argument(
        '--log', type=bool, default=False,
        help='saving logs')
    parser.add_argument(
        '--num_workers', type=int, default=3,
        help='number of process you have')


    #TRAIN OPTION BY NN
    mode=parser.parse_known_args(args)[0].mode.lower()
    if 'kd' in mode:
        if 'offkd' in mode:
            parser.add_argument('--pretrained_model',type=str,default='vgg16',help='set pretrained_model')
            parser.add_argument('--temperature',type=float,default=1.0,help='default:softmax')
            parser.add_argument('--kd_type',type=str,default='softtarget',help='default:softtarget')
        elif 'onkd' in mode:
            parser.add_argument('--kd_type',type=str,default='softtarget',help='default:softtarget')
        elif 'ensemblekd' in mode:
            parser.add_argument('--kd_type',type=str,default='dml',help='default:dml')
            parser.add_argument('--num_model',type=int,default=2,help='default:2')
        else:
            raise NotImplementedError
        kd=True
    else:
        kd=False
    model = parser.parse_known_args(args)[0].model.lower()
    from Model.baseNet import get_hyperparams
    dataset,epochs,lr,momentum=get_hyperparams(model,kd)

    parser.add_argument(
        '--lr', type=float, default=lr,
        help='set learning rate')
    parser.add_argument(
        '--momentum', type=float, default=momentum,
        help='set learning rate')
    parser.add_argument(
        '--epochs', type=int, default=epochs,
        help='run epochs')
    parser.add_argument(
        '--start_epoch', type=int, default=1,help='for load model'
    )
    parser.add_argument(
        '--dataset', type=str, default=dataset,
        help='choose dataset, if nn==lenet5,mnist elif nn==vgg16,cifar10')
    parser.add_argument(
        '--file_name', type=str, default=None,
        help='replay name')
    parser.add_argument(
        '--earlystop', type=bool, default=False,
        help='earlystopping')
    parser.add_argument(
        '--patience', type=int, default=10,
        help='set mini-batch size')

        
    return parser.parse_known_args(args)[0]


def main(args):
    flags = parse_args(args)
    TRAIN_MODE=['train','train_offkd','train_ensemblekd','train_onkd']
    train_mode_list=TRAIN_MODE

    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),'training_data')) == False:
        os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),'training_data'))
    if flags.file_name is None and flags.mode in train_mode_list:
        time_data = time.strftime(
            '%m-%d_%H-%M-%S', time.localtime(time.time()))
        print(time_data)
    elif flags.file_name is not None and flags.mode not in train_mode_list:  # load param when not training
        time_data = flags.file_name
        file_name = flags.file_name
    else:
        file_name = None  # no file name just read from grad.csv, .npy and .pt
    
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda and flags.device == 'gpu' else "cpu")
    # Random Seed 설정
    random_seed = flags.seed
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.random.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    '''
    Basic Setting
    '''
    configs=vars(flags)
    configs ['device']=str(device)

    if configs['mode'] in train_mode_list:
        save_params(configs, time_data)
        print("Using device: {}, Mode:{}, Type:{}".format(device,flags.mode,flags.model))
        # sys.stdout=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'training_data','log_{}.txt'.format(time_data)),'w')
    else:
        if flags.file_name is not None:
            CALL_CONFIG = load_params(configs, file_name)
            CALL_CONFIG['visual_mode']=CALL_CONFIG['mode'] # training종류 선택
            CALL_CONFIG['mode']=configs['mode']
            if configs['mode']=='visual':
                CALL_CONFIG['visual_type']=configs['visual_type']
                print(configs['visual_type'])
            configs=CALL_CONFIG
            print("Mode:{}, Type:{}".format(configs['mode'],configs['model']))
    
    
    #Train
    file_path=os.path.dirname(os.path.abspath(__file__))
    if flags.mode == 'train':
        model=BaseNet(configs).model
        from Learner.baselearner import ClassicLearner
        learner=ClassicLearner(model,time_data,file_path,configs)
        learner.run()
    elif flags.mode=='train_offkd':
        configs=dict(configs ,**{'pretrained_model':flags.pretrained_model,'temperature':flags.temperature,'kd_type':flags.kd_type})
        base_model=BaseNet(configs)
        model=base_model.model
        pretrained_model=base_model.pretrained_model
        from Learner.offkd_learner import OFFKDLearner
        learner=OFFKDLearner(model,pretrained_model,time_data,file_path,configs)
        learner.run()
    elif flags.mode=='train_ensemblekd':
        model_list=list()
        for _ in range(configs['num_model']):
            model_list.append(BaseNet(configs).model)
        from Learner.ensemble_learner import EnsembleLearner
        learner=EnsembleLearner(model_list,time_data,file_path,configs)
        learner.run()
    
    print("End the process")

if __name__ == '__main__':
    main(sys.argv[1:])
