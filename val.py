import argparse
import os
from glob import glob

import PIL.Image
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision.models.segmentation import fcn_resnet50
import yaml
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei'] # Or any other Chinese charactersimport albumentations
import albumentations
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter


def load_model(model_name):
    with open('models/%s/config.yml' % model_name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)
    cudnn.benchmark = True
    # create model
    print("=> creating model %s" % config['arch'])
    if config['arch'] == 'FCN':
        model = fcn_resnet50(num_classes=config['num_classes'], aux_loss=False)  # no pre-trained weights are used.
    else:
        model = archs.__dict__[config['arch']](config['num_classes'],
                                               config['input_channels'],
                                               deep_supervision=config['deep_supervision'])
    # model = model.cuda()
    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()
    return config, model


# config, model = load_model('矿石图像分割_NestedUNet_woDS')
val_transform = Compose([
        albumentations.Resize(512, 512),
        albumentations.Normalize(),
        ])


def compute_an_image(filename):
    img = np.expand_dims(np.array(Image.open(filename)), -1)
    img_ = np.repeat(img, 3, -1)
    img = val_transform(image=img_)
    img = np.transpose(img['image'], [2, 0, 1])[:1, :, :].astype(float) / 255

    with torch.no_grad():
            print(f'Start make inference om image {filename}')
            # img = torch.tensor(img).cuda().unsqueeze(0).float()
            img = torch.tensor(img).unsqueeze(0).float()

            # compute output
            if config['deep_supervision']:
                output = model(img)[-1]
            else:
                output = model(img)

            output = torch.sigmoid(output)

            # for i in range(len(output)):
            #     for c in range(config['num_classes']):
            #         export_path = os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg')
            #         img_array = (output[i, c] * 255).astype('uint8')
            #         img = Image.fromarray(img_array)
            #         img.save(export_path)
            pred_mask = np.zeros([config['num_classes'], config['input_h'], config['input_w']], dtype=np.uint8)
            pred_classes = output.squeeze().argmax(dim=0).cpu().numpy()
            for class_id in range(config['num_classes']):
                color_mask = pred_classes == class_id
                color_mask = np.where(color_mask, 255, 0)
                pred_mask[class_id] = color_mask

            fig, ax = plt.subplots(1, 2)
            plt.suptitle('矿石分割结果')
            ax[0].imshow(img_)
            ax[0].set_title('原图')
            ax[1].imshow(pred_mask.transpose([1, 2, 0]))
            ax[1].set_title('分割结果')

    print('Inference completes')
    return fig


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    model_name = args.name

    config, model = load_model(model_name)

    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'JPEGImages', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    is_single_channel = 'FCN' not in model_name
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'JPEGImages'),
        mask_dir=os.path.join('inputs', config['dataset'], 'Annotations'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform,
        is_single_channel=is_single_channel)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    with torch.no_grad():
        batch_counter = 0
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            # input = input.cuda()
            # target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
                if not isinstance(output, torch.Tensor):  # FCN has different output format
                    output = output['out']
            else:
                output = model(input)
                if not isinstance(output, torch.Tensor):  # FCN has different output format
                    output = output['out']

            iou = iou_score(output, target)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()
            output_path = os.path.join('outputs', config['name'], 'batch_{}.npy'.format(batch_counter))
            np.save(output_path, output)
            batch_counter += 1

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    export_path = os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.jpg')
                    img_array = (output[i, c] * 255).astype('uint8')
                    img = Image.fromarray(img_array)
                    img.save(export_path)

    print('IoU: %.4f' % avg_meter.avg)

    # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
