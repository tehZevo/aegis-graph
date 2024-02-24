from uuid import uuid4

class Source:
    def __init__(self):
        #TODO: save this?
        self.id = uuid4()
    
    def get_state(self):
        raise NotImplementedError