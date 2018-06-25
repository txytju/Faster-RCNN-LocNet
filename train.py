import os

import numpy as np
import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

from model.utils.bbox_tools import bbox_iou


import resource

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000, prob_thre=0.7):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        # imgs here are reshaped images
        # sizes here are the original shape of images
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes], prob_thre=prob_thre)
        
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)

    # data
    print('load data')
    dataloader = data_.DataLoader(dataset, 
                                  batch_size=1, 
                                  shuffle=False, 
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, 
                                       pin_memory=True
                                       )

    # model and trainer
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')

    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)

            # print(label)
            
            # all the input data for one training are : img, bbox, label, scale
            trainer.train_step(img, bbox, label, scale)
            # training code stop here.


            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)

                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        
        # use the test dataset to eval
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num, prob_thre=opt.prob_thre)

        print("eval_result", eval_result)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        trainer.vis.plot('test_map', eval_result['map'])
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)
        if epoch == 13: 
            break



def eval_prob_thre(**kwargs):
    '''
    Use the best trained model to find out the best prob_thre, \
    which is used when generating prediction box using px and py.
    '''
    opt._parse(kwargs)

    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, 
                                       pin_memory=True
                                       )

    # model and trainer
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')

    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    best_map = 0
    best_prob_thre = 0
    
    for prob_thre in np.linspace(0.3,0.9,7):
        
        # use the test dataset to eval
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num, prob_thre=prob_thre)
        print("eval_result", eval_result)
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_prob_thre = prob_thre

    print("best_map is ", best_map)
    print("best prob_thre is ", best_prob_thre)


def eval_on_question_dataset(iou_thre=0.8):
    '''
    Use the best trained model to predict on question dataset and evaluate the percision and accuracy.
    '''
    # opt._parse(kwargs)

    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, 
                                       pin_memory=True)

    test_num = len(test_dataloader) # use all data in testset to caculate metrics

    # model and trainer
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')

    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    
    for prob_thre in np.linspace(0.3,0.9,7):

        num_TP = 0
        num_FP = 0
        num_GTBB = 0
        
        # get predicted anchor boxes and GTBBs
        for ii, (imgs, sizes, gt_bboxes_, _, _) in tqdm(enumerate(test_dataloader)):
            
            sizes = [sizes[0][0], sizes[1][0]]
            pred_bboxes_, _, _ = faster_rcnn.predict(imgs, [sizes], prob_thre=prob_thre) 
            # pred_bboxes_ and gt_bboxes_ are both in the raw image size.
            # both like [ymin, xmin, ymax, xmax]

            gt_bboxes = np.squeeze(at.tonumpy(gt_bboxes_), axis=0)
            pred_bboxes = at.tonumpy(pred_bboxes_[0])
            
            ious = bbox_iou(gt_bboxes, pred_bboxes)

            best_iou = np.max(ious, axis=1)
            tp = best_iou > iou_thre
            
            num_TP += sum(tp)
            num_FP += pred_bboxes.shape[0] - sum(tp)
            num_GTBB += gt_bboxes.shape[0]

            if ii == test_num: break
        print("prob_thre = ", prob_thre)
        print("Percission = ", num_TP / (num_TP + num_FP))
        print("Accuracy = ", num_TP / num_GTBB)   



if __name__ == '__main__':
    import fire

    fire.Fire()
