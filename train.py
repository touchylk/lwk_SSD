# coding: utf-8
import argparse
import logging
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from ssd.engine.trainer import do_train
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
from ssd.utils.misc import str2bool


def train(cfg, args):
    logger = logging.getLogger('SSD.trainer')
    model = build_detection_model(cfg) # 建立模型
    device = torch.device(cfg.MODEL.DEVICE) # 看cfg怎么组织的，把文件和args剥离开
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
        # model = nn.DataParallel(model)

    lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    optimizer = make_optimizer(cfg, model, lr) # 建立优化器

    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)

    arguments = {"iteration": 0}
    save_to_disk = dist_util.get_rank() == 0
    checkpointer = CheckPointer(model, optimizer, scheduler, save_dir=cfg.OUTPUT_DIR, save_to_disk=save_to_disk, logger=logger)
    # 建立模型存储载入类,给save_dir赋值表示
    extra_checkpoint_data = checkpointer.load(f='', use_latest=False) # 载入模型
    arguments.update(extra_checkpoint_data)

    max_iter = cfg.SOLVER.MAX_ITER // args.num_gpus
    train_loader = make_data_loader(cfg, is_train=True, distributed=args.distributed, max_iter=max_iter, start_iter=arguments['iteration']) # 建立数据库

    print("dataloader: ",train_loader.batch_size)
    # exit(1232)
    model = do_train(cfg, model, train_loader, optimizer, scheduler, checkpointer, device, arguments, args) # 训练
    return model

def write_cfg_args(cfg,args):
    config_args_file = os.path.join(cfg.OUTPUT_DIR, 'config_args.txt')
    with open(config_args_file, 'w') as f:
        f.write('config:\n\n')
        f.write(str(cfg))
        f.write('\n\nargs:\n\n')
        f.write(str(args))



def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    config_file = './configs/mobilenet_v2_ssd320_voc0712.yaml'
    # parser.add_argument(
    #     "--config-file",
    #     default="",
    #     metavar="FILE",
    #     help="path to config file",
    #     type=str,
    # )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--log_step', default=10, type=int, help='Print logs every log_step')
    parser.add_argument('--save_step', default=2500, type=int, help='Save checkpoint every save_step')
    parser.add_argument('--eval_step', default=2500, type=int, help='Evaluate dataset every eval_step, disabled when eval_step < 0')
    parser.add_argument('--use_tensorboard', default=True, type=str2bool)
    parser.add_argument('--num_gpus',default=2,type=int)
    parser.add_argument("--skip-test", dest="skip_test", help="Do not test the final model", action="store_true",)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    args.num_gpus = num_gpus
    # num_gpus = args.num_gpus
    # args.distributed = num_gpus > 1

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # print(cfg)
    # exit()
    # print(cfg.INPUT)
    # exit()
    print(cfg.OUTPUT_DIR)

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    # logger.info("Loaded configuration file {}".format(config_file))
    # with open(args.config_file, "r") as cf:
    #     config_str = "\n" + cf.read()
    #     logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    write_cfg_args(cfg,args)
    model = train(cfg, args)

    if not args.skip_test:
        logger.info('Start evaluating...')
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        do_evaluation(cfg, model, distributed=args.distributed)


if __name__ == '__main__':
    main()
