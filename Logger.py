import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

class Logger(SummaryWriter):
    def add_scalar_dict(self, scalar_dict, global_step= None, walltime= None, display_name= '', summary_description= ''):
        for tag, scalar in scalar_dict.items():
            self.add_scalar(
                tag= tag,
                scalar_value= scalar,
                global_step= global_step,
                walltime= walltime,
                display_name= display_name,
                summary_description= summary_description
                )

    def add_image_dict(self, image_dict, global_step, walltime= None):
        for tag, data in image_dict.items():
            fig= plt.figure(figsize=(10, 5), dpi= 100)
            if data.ndim == 1:
                plt.imshow([[0]], aspect='auto', origin='lower')
                plt.plot(data)
                plt.margins(x= 0)
            elif data.ndim == 2:
                plt.imshow(data, aspect='auto', origin='lower')
            plt.colorbar()
            plt.title(tag)
            plt.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            self.add_image(tag= tag, img_tensor= data, global_step= global_step, walltime= walltime, dataformats= 'HWC')

    def add_histogram_model(self, model, global_step=None, bins='tensorflow', walltime=None, max_bins=None, delete_keywords= []):
        for tag, parameter in model.named_parameters():
            x = tag
            tag = '/'.join([x for x in tag.split('.') if not x in delete_keywords])

            self.add_histogram(
                tag= tag,
                values= parameter.data.cpu().numpy(),
                global_step= global_step,
                bins= bins,
                walltime= walltime,
                max_bins= max_bins
                )