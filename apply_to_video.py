import numpy as np
import torch
import imageio
from PIL import Image
from tqdm import tqdm

from utils import utils, warp, image_utils, constant_var
from models import end_2_end_optimization
from options import fake_options
from skimage import img_as_ubyte

# if want to run on CPU, please make it False
constant_var.USE_CUDA = True
utils.fix_randomness()

# if GPU is RTX 20XX, please disable cudnn
torch.backends.cudnn.enabled = True


# set some options
opt = fake_options.FakeOptions()
opt.batch_size = 32
opt.coord_conv_template = True
opt.error_model = 'loss_surface'
opt.error_target = 'iou_whole'
opt.guess_model = 'init_guess'
opt.homo_param_method = 'deep_homography'
opt.load_weights_error_model = 'pretrained_loss_surface'
opt.load_weights_upstream = 'pretrained_init_guess'
opt.lr_optim = 1e-5
opt.need_single_image_normalization = True
opt.need_spectral_norm_error_model = True
opt.need_spectral_norm_upstream = False
opt.optim_criterion = 'l1loss'
opt.optim_method = 'stn'
opt.optim_type = 'adam'
opt.out_dir = './out'
opt.prevent_neg = 'sigmoid'
opt.template_path = './data/world_cup_template.png'
opt.warp_dim = 8
opt.warp_type = 'homography'


opt.goal_image_path = 'demo_video_frame.jpg'
# opt.goal_image_path = './data/world_cup_2018.png'
opt.optim_iters = 80


e2e = end_2_end_optimization.End2EndOptimFactory.get_end_2_end_optimization_model(opt)


# template_image
template_image = imageio.imread(opt.template_path, pilmode='RGB')
template_image = template_image / 255.0
if opt.coord_conv_template:
    template_image = image_utils.rgb_template_to_coord_conv_template(template_image)
# covert np image to torch image, and do normalization
template_image = utils.np_img_to_torch_img(template_image)
if opt.need_single_image_normalization:
    template_image = image_utils.normalize_single_image(template_image)

template_image_draw = imageio.imread(opt.template_path, pilmode='RGB')
template_image_draw = template_image_draw / 255.0
template_image_draw = image_utils.rgb_template_to_coord_conv_template(template_image_draw)
template_image_draw = utils.np_img_to_torch_img(template_image_draw)


reader = imageio.get_reader('demo_video_trim.mp4')
fps = reader.get_meta_data()['fps']
frame_list = []
for im in reader:
    frame_list.append(im)
reader.close()
frame_shape = frame_list[0].shape[:2]


first_frame = True
optim_homography_list = []
for idx, frame in tqdm(enumerate(frame_list), total=len(frame_list)):
    pil_image = Image.fromarray(np.uint8(frame))
    pil_image = pil_image.resize([256, 256], resample=Image.NEAREST)
    frame = np.array(pil_image)
    frame = utils.np_img_to_torch_img(frame)
    if opt.need_single_image_normalization:
        frame = image_utils.normalize_single_image(frame)
    _, optim_homography = e2e.optim(frame[None], template_image, refresh=first_frame)
    optim_homography_list.append(optim_homography.detach())
    first_frame = False


warped_tmp_optim_list = []
for optim_h in optim_homography_list:
    warped_tmp_optim = warp.warp_image(template_image_draw, optim_h, out_shape=frame_shape)[0]
    warped_tmp_optim = utils.torch_img_to_np_img(warped_tmp_optim)
    warped_tmp_optim_list.append(warped_tmp_optim)


# generating result video
result = []
edge_color = [0, 1.0, 0]
for frame, template in tqdm(zip(frame_list, warped_tmp_optim_list), total=len(frame_list)):
    video_content = frame / 255.0
    template_content = template
    valid_index = template_content[..., 0] > 0
    edge_index = template_content[..., 0] >= 254.0 / 255.0
    overlay = (video_content[valid_index].astype('float32') + template_content[valid_index].astype('float32')) / 2
    out_frame = video_content.copy()
    out_frame[valid_index] = overlay
    out_frame[edge_index] = edge_color
    out_frame = out_frame * 255.0
    out_frame = out_frame.astype('uint8')
    result.append(out_frame)
imageio.mimsave("result.mp4", [img_as_ubyte(p) for p in result], fps=fps)
