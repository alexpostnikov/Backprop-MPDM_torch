import torch
import logging


class Utils:
    def __init__(self):
        pass

    def check_poses_not_the_same(self, pose1, pose2, grad1, grad2, lr):
        counter = 100
        while torch.norm(pose1 - pose2) < 0.6 and counter:
            pose1 = pose1 - lr/4. * grad1
            pose2 = pose2 - lr/4. * grad2
            counter -= 1
        return pose1, pose2


    def setup_logger(self, logger_name, log_file = 'logs/log.log'):
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            
            fh = logging.FileHandler('logs/log.log')
            fh.setLevel(logging.INFO)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            
            logger.addHandler(fh)
            logger.addHandler(ch)

            logger.info("----------------------starting script------------------------------------")
            return logger
            ####### logging init end ######