r""" Logging during training/testing """
import logging
import os

from tensorboardX import SummaryWriter
from datetime import datetime
import torch
from .utils import is_main_process, save_on_master, reduce_metric


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self):
        self.J_buf = torch.zeros([1]).float().cuda()
        self.F_buf = torch.zeros([1]).float().cuda()
        self.count = torch.zeros([1]).float().cuda()
        self.loss_buf = []

    def update(self, J, F, loss):
        self.J_buf += J.sum()
        self.F_buf += F.sum()
        self.count += J.shape[0]
        if loss is None:
            loss = torch.tensor(0.0)
        self.loss_buf.append(loss)

    def compute_jf(self):
        jm = self.J_buf / self.count
        fm = self.F_buf / self.count
        jf = (jm + fm) / 2

        return jf, jm, fm

    def write_result(self, split, epoch, print_res=False):
        self.J_buf, self.F_buf, self.count = self.reduce_metrics([self.J_buf, self.F_buf, self.count], False)
        JF, J, F = self.compute_jf()

        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
        msg = '\n*** %s ' % split
        msg += '[%s] ' % formatted_now
        msg += '[@Epoch %02d] ' % epoch if epoch != -1 else ''
        if epoch != -1:
            loss_buf = torch.stack(self.loss_buf)
            loss_buf = self.reduce_metrics([loss_buf])[0]
            msg += 'Avg L: %6.5f  ' % loss_buf.mean()
        msg += 'J&F: %5.2f   ' % JF
        msg += 'J: %5.2f   ' % J
        msg += 'F: %5.2f   ' % F

        msg += '***\n'
        if print_res:
            print(msg)
        Logger.info(msg)

    def write_process(self, batch_idx, datalen, epoch, write_batch_idx=20):
        if batch_idx % write_batch_idx == 0:
            msg = '[Epoch: %02d] ' % epoch if epoch != -1 else ''
            msg += '[Batch: %04d/%04d] ' % (batch_idx+1, datalen)
            JF, J, F = self.compute_jf()
            if epoch != -1:
                loss_buf = torch.stack(self.loss_buf)
                msg += 'L: %6.5f  ' % loss_buf[-1]
                msg += 'Avg L: %6.5f  ' % loss_buf.mean()
            msg += 'J&F: %5.2f   ' % JF
            msg += 'J: %5.2f   ' % J
            msg += 'F: %5.2f   ' % F
            Logger.info(msg)
    def reduce_metrics(self, metrics, average=True):
        reduced_metrics = []
        for m in metrics:
            reduce_metric(m, average)
            reduced_metrics.append(m)
        return reduced_metrics


class Logger:
    r""" Writes evaluation results of training/testing """
    @classmethod
    def initialize(cls, args, training):
        logtime = datetime.now().__format__('_%m%d_%H%M%S')
        logpath = args.logpath if training else '_TEST_' + args.load.split('/')[-2].split('.')[0] + logtime
        if logpath == '': logpath = logtime

        cls.logpath = os.path.join('logs', logpath)
        cls.benchmark = args.benchmark
        os.makedirs(cls.logpath, exist_ok=True)

        logging.basicConfig(filemode='w',
                            filename=os.path.join(cls.logpath, 'log.txt'),
                            level=logging.INFO,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M:%S')

        # Console log config
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

        # Tensorboard writer
        cls.tbd_writer = SummaryWriter(os.path.join(cls.logpath, 'tbd/runs'))

        # Log arguments
        logging.info('\n:=========== In-Context Seg. with DC-SAM ===========')
        for arg_key in args.__dict__:
            logging.info('| %20s: %-24s' % (arg_key, str(args.__dict__[arg_key])))
        logging.info(':====================================================\n')

    @classmethod
    def info(cls, msg):
        r""" Writes log message to log.txt """
        logging.info(msg)

    @classmethod
    def save_model_jf(cls, model, epoch, val_JF):
        torch.save(model.state_dict(), os.path.join(cls.logpath, 'best_model.pt'))
        cls.info('Model saved @%d w/ val. J&F: %5.2f.\n' % (epoch, val_JF))

    @classmethod
    def log_params(cls, model):
        backbone_param = 0
        learner_param = 0
        total_param = 0
        for k in model.state_dict().keys():
            n_param = model.state_dict()[k].view(-1).size(0)
            if [i for i in ['layer0', 'layer1', 'layer2', 'layer3', 'layer4', 'dinov2'] if i in k]:
                backbone_param += n_param
        for param in model.parameters():
            if param.requires_grad:
                learner_param += param.numel()
            total_param += param.numel()
        Logger.info('Backbone # param.: %d' % backbone_param)
        Logger.info('Learnable # param.: %d' % learner_param)
        Logger.info('Total # param.: %d' % total_param)

