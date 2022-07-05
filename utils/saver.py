import os
import torchvision
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image

class Saver():
    def __init__(self, display_dir,display_freq):
        self.display_dir = display_dir
        self.display_freq = display_freq
        # make directory
        if not os.path.exists(self.display_dir):
            os.makedirs(self.display_dir)
        # create tensorboard writer
        self.writer = SummaryWriter(logdir=self.display_dir)

  # write losses and images to tensorboard
    def write_display(self, total_it, loss, image=None, force_write=False):
        if force_write or (total_it + 1) % self.display_freq == 0:
            # write img
            if image is not None:
                for m in image:
                    image_dis = torchvision.utils.make_grid(image[m].detach().cpu(), nrow=5)/2 + 0.5
                    self.writer.add_image(m, image_dis, total_it)
            for l in loss:
                self.writer.add_scalar(l[0], l[1], total_it)
                print(l[0], l[1], total_it)
