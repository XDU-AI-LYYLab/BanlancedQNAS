import torch
import torch.nn as nn
import torch.nn.functional as F


def get_lr_scheduler(
        optimizer,
        lr_schedule,
        logger,
        total_epochs=None,
        step_per_epoch=None,
        step_size=None,
        decay_ratio=None):
    """
    Return learning rate scheduler for optimizer
    """
    logger.info("================ Scheduler =================")
    logger.info("Scheduler : {}".format(lr_schedule))

    if lr_schedule == "cosine":
        assert step_per_epoch is not None
        assert total_epochs is not None

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=step_per_epoch * total_epochs)

    elif lr_schedule == "step":
        assert step_size is not None
        assert decay_ratio is not None

        logger.info("Step size (epoch) : {}".format(step_size))
        logger.info("Gamma : {}".format(decay_ratio))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size * step_per_epoch,
            gamma=decay_ratio,
            last_epoch=-1)

    return lr_scheduler


def get_optimizer(
        model_parameters,
        optimizer_type,
        learning_rate,
        weight_decay,
        logger,
        momentum=None,
        alpha=None,
        beta=None):
    logger.info("================= Optimizer =================")
    logger.info("Optimizer : {}".format(optimizer_type))
    logger.info("Learning rate : {}".format(learning_rate))
    logger.info("Weight decay : {}".format(weight_decay))

    if optimizer_type == "sgd":
        assert momentum is not None

        logger.info("Momentum : {}".format(momentum))
        optimizer = torch.optim.SGD(params=model_parameters,
                                    lr=learning_rate,
                                    momentum=momentum,
                                    weight_decay=weight_decay)

    elif optimizer_type == "rmsprop":
        assert momentum is not None

        logger.info("Momentum : {}".format(momentum))
        optimizer = torch.optim.RMSprop(model_parameters,
                                        lr=learning_rate,
                                        alpha=alpha,
                                        momentum=momentum,
                                        weight_decay=weight_decay)
    elif optimizer_type == "adam":
        assert beta is not None

        logger.info("Beta : {}".format(beta))
        optimizer = torch.optim.Adam(model_parameters,
                                     weight_decay=weight_decay,
                                     lr=learning_rate,
                                     betas=(beta, 0.999))

    return optimizer


def get_criterion():
    return cross_entropy_with_label_smoothing


def get_hc_criterion():
    return l2_hc_loss


def cross_entropy_with_label_smoothing(pred, target, eta=0.1):
    onehot_target = label_smoothing(pred, target, eta=eta)
    return cross_entropy_for_onehot(pred, onehot_target)


def label_smoothing(pred, target, eta=0.1):
    """
    Reference : https://arxiv.org/pdf/1512.00567.pdf
    """
    n_class = pred.size(1)
    target = torch.unsqueeze(target, 1)

    onehot_target = torch.zeros_like(pred)
    onehot_target.scatter_(1, target, 1)
    return onehot_target * (1 - eta) + eta / n_class * 1


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def l2_hc_loss(search_hc, target_hc, hc_weight):
    return (search_hc - target_hc) ** 2 * hc_weight
