import torch
import torch.nn as nn

class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sketch, view=None):
        # Only include 'view' in the input dict if it's provided. Passing None
        # caused set_input to try calling .to() on None leading to AttributeError.
        input_dict = {'image': sketch}
        if view is not None:
            input_dict['view'] = view
        self.model.set_input(input_dict)
        self.model.forward_inference()
        return self.model.out_vertices, self.model.out_faces

class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, z, class_id):
        # The decoder from view_disentangle_model is more complex
        # and tied to the encoder. This is a simplified placeholder.
        # We might need to adjust this based on how the FewShotScorer
        # uses the decoder output.
        # For now, let's assume it returns a mesh.
        return self.model.out_vertices
