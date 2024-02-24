import numpy as np

from .sensor import Sensor

class ManualSensor(Sensor):
    def __init__(self, size):
        super().__init__()
        self.state = np.zeros([size])
    
    def set_state(self, state):
        self.state = state
    
    def get_state(self):
        return self.state