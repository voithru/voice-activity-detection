from torch.optim.lr_scheduler import CyclicLR, ExponentialLR, ReduceLROnPlateau

from vad.lr_scheduling.configs import LRSchedulerConfig

from .constant import Constant
from .cosine_annealing_with_warmup import CosineAnnealingWarmUpRestarts
from .inverse_squre import InverseSquareRootSchedule
from .noam import Noam
from .rampup_hold_decay import RampUpHoldDecay


def select_lr_scheduler(optimizer, lr_scheduler_config: LRSchedulerConfig):
    if lr_scheduler_config is None:
        lr_scheduler = Constant(optimizer=optimizer)
    elif lr_scheduler_config.name == "ramp-up-hold-decay":
        lr_scheduler = RampUpHoldDecay(
            optimizer=optimizer,
            high_plateau_lr=lr_scheduler_config.ramp_up_hold_decay.high_plateau_lr,
            ramp_up_milestone=lr_scheduler_config.ramp_up_hold_decay.ramp_up_milestone,
            hold_milestone=lr_scheduler_config.ramp_up_hold_decay.hold_milestone,
            decay_milestone=lr_scheduler_config.ramp_up_hold_decay.decay_milestone,
        )
    elif lr_scheduler_config.name == "noam":
        lr_scheduler = Noam(
            optimizer=optimizer,
            factor=lr_scheduler_config.noam.factor,
            d_model=lr_scheduler_config.noam.d_model,
            warmup_steps=lr_scheduler_config.noam.warmup_steps,
        )
    elif lr_scheduler_config.name == "cyclical":
        lr_scheduler = CyclicLR(
            optimizer=optimizer,
            base_lr=1e-6,
            max_lr=lr_scheduler_config.cyclic.max_lr,
            step_size_up=lr_scheduler_config.cyclic.step_size_up,
            step_size_down=lr_scheduler_config.cyclic.step_size_down,
            mode=lr_scheduler_config.cyclic.mode,
            gamma=lr_scheduler_config.cyclic.gamma,
            cycle_momentum=False,
        )
    elif lr_scheduler_config.name == "cosine":
        lr_scheduler = CosineAnnealingWarmUpRestarts(
            optimizer=optimizer,
            update_steps=lr_scheduler_config.cosine_annealing.update_steps,
            step_multiple=lr_scheduler_config.cosine_annealing.step_multiple,
            max_annealing_lr=lr_scheduler_config.cosine_annealing.max_lr,
            annealing_warmup_steps=lr_scheduler_config.cosine_annealing.warmup_steps,
            gamma=lr_scheduler_config.cosine_annealing.gamma,
        )
    elif lr_scheduler_config.name == "exponential":
        lr_scheduler = ExponentialLR(
            optimizer=optimizer, gamma=lr_scheduler_config.exponential.gamma
        )

    elif lr_scheduler_config.name == "inverse-sqrt":
        lr_scheduler = InverseSquareRootSchedule(
            optimizer=optimizer,
            max_lr=lr_scheduler_config.inverse_squre.max_lr,
            warmup_init_lr=lr_scheduler_config.inverse_squre.warmup_init_lr,
            warmup_updates=lr_scheduler_config.inverse_squre.warmup_steps,
        )
    elif lr_scheduler_config.name == "reduce-on-plateau":
        lr_scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            factor=lr_scheduler_config.reduce_on_plateau.factor,
            patience=lr_scheduler_config.reduce_on_plateau.patience,
            verbose=True,
            threshold=lr_scheduler_config.reduce_on_plateau.threshold,
            threshold_mode=lr_scheduler_config.reduce_on_plateau.threshold_mode,
        )
    else:
        raise NotImplementedError

    return lr_scheduler
