import logging
logger = logging.getLogger('base')


def create_model(opt, dp_device_ids, local_rank):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt, dp_device_ids, local_rank)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
