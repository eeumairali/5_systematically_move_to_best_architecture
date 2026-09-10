




class BackBone:
    """backbone for JDE model"""
    def __init__(self):
        self.backbone = None
    def make_backbone(self):
        raise NotImplementedError("make_backbone method should be implemented in subclass")





class Architecture(BackBone):
    """main architecture for JDE model"""
    def __init__(self):
        super(Architecture, self).__init__()
        self.make_backbone()

    def make_neck(self):
        raise NotImplementedError("make_neck method should be implemented in subclass")
    def make_head_detection(self):
        raise NotImplementedError("make_head_detection method should be implemented in subclass")
    def make_head_embedding(self):
        raise NotImplementedError("make_head_embedding method should be implemented in subclass")

    


class JDEModel(Architecture):
    """JDE model with backbone, neck, detection head and embedding head"""
    def __init__(self):
        super(JDEModel, self).__init__()
        self.make_neck()
        self.make_head_detection()
        self.make_head_embedding()

    def forward(self, x):
        raise NotImplementedError("forward method should be implemented in subclass")
    def backpropogation(self, loss):
        raise NotImplementedError("backpropogation method should be implemented in subclass")
    def inference(self, x):
        raise NotImplementedError("inference method should be implemented in subclass")


method should be implemented in subclass")


