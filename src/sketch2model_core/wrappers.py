import torch
import torch.nn as nn
from .SoftRas import soft_renderer as sr

class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, sketch, view=None):
        # Only include 'view' in the input dict if it's provided.
        input_dict = {'image': sketch}
        if view is not None:
            input_dict['view'] = view
        self.model.set_input(input_dict)
        self.model.forward_inference()
        return self.model.out_zs, self.model.out_zv_pred

class DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, zs, class_id, zv):
        # Combines decoding and rendering to produce a silhouette from latents.
        # class_id is currently unused but kept for API consistency.

        # 1. Get final latent z from shape (zs) and view (zv) latents.
        z = self.model.netFull.decoder(zs, zv)
        
        # 2. Decode mesh from z
        vertices, faces = self.model.netFull.mesh_decoder(z)

        # 3. Render the silhouette to compare with the input sketch.
        view_angles_norm = self.model.netFull.view_decoder(zv)
        view_angles = self.model.decode_view(view_angles_norm)
        camera = self.model.view2camera(view_angles)
        
        # Use the primary, full-resolution renderer from the model
        renderer = self.model.renderers[0]
        
        renderer.to(vertices.device)

        transform = sr.LookAt(viewing_angle=15, eye=camera)
        mesh = sr.Mesh(vertices, faces)
        
        silhouettes = renderer(transform(mesh))
        
        # The renderer output is RGBA, we only need the alpha channel for the silhouette
        silhouette = silhouettes[:, 3:4, :, :]

        return silhouette
