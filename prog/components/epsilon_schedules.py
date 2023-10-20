"""Stores the exploration schedules."""

import numpy as np


class DecayThenFlatSchedule():
    """Epsilon Decay Schedule.
    
    Attributes:
        start: starting value of the schedule.
        finish: end value of the schedule.
        time_length: length of the schedule.
        delta: change per schedule step.
        decay: decay parameter of the schedule.
    """

    def __init__(self,
                 start,
                 finish,
                 time_length,
                 decay="exp"):
        """Initialize the schedule."""

        self.start = start
        self.finish = finish
        self.time_length = time_length
        self.delta = (self.start - self.finish) / self.time_length
        self.decay = decay

        if self.decay in ["exp"]:
            self.exp_scaling = (-1) * self.time_length /
            np.log(self.finish) if self.finish > 0 else 1

    def eval(self, T):
        """Get the current step schedule value.
        
        Args:
            T: current schedule step.
        """
        if self.decay in ["linear"]:
            return max(self.finish, self.start - self.delta * T)
        elif self.decay in ["exp"]:
            return min(self.start, max(self.finish, np.exp(- T / self.exp_scaling)))
    pass
