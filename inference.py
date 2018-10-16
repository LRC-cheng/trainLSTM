import os
import time
import codecs
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.DEBUG,
                    format = '%(asctime)s[%(levelname)s] ---- %(message)s',
                    )

import torch
import torch.utils.data as Data

from read_data import get_samples, get_data, TorchDataSet
from net_component import LanNet

## ======================================
# 配置文件和参数
# 数据列表
train_list = "./label_train_list_fb.txt"
dev_list   = "./label_dev_list_fb.txt"

# 基本配置参数
use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda:0")

# 保存模型地址
model_dir = "./inference/models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
# 网络参数
dimension = 40
language_nums =6
learning_rate = 0.1
batch_size = 64
chunk_num = 10
train_iteration = 1
display_fre = 50
half = 4
half_1=7
epoch=0
## ======================================
train_dataset = TorchDataSet(train_list, batch_size, chunk_num, dimension)
dev_dataset = TorchDataSet(dev_list, batch_size, chunk_num, dimension)
logging.info('finish reading all train data')

# 优化器，SGD更新梯度
train_module = LanNet(input_dim=dimension, hidden_dim=128, bn_dim=30, output_dim=language_nums)
logging.info(train_module)
#optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
train_module.load_state_dict(torch.load('./inference/model9.model', map_location=lambda storage, loc: storage))
#optimizer = torch.optim.RMSprop(train_module.parameters(), lr=learning_rate,  alpha=0.9)
optimizer = torch.optim.SGD(train_module.parameters(), lr=learning_rate, momentum=0.9)
# 将模型放入GPU中
if use_cuda:
    train_module = train_module.to(device)


train_module.eval()
epoch_tic = time.time()
dev_loss = 0.
dev_acc = 0.
dev_batch_num = 0

for step, (batch_x, batch_y) in enumerate(dev_dataset):
    tic = time.time()

    batch_target = batch_y[:, 0].contiguous().view(-1, 1).long()
    batch_frames = batch_y[:, 1].contiguous().view(-1, 1)

    max_batch_frames = int(max(batch_frames).item())
    batch_dev_data = batch_x[:, :max_batch_frames, :]

    step_batch_size = batch_target.size(0)
    batch_mask = torch.zeros(step_batch_size, max_batch_frames)
    for ii in range(step_batch_size):
        frames = int(batch_frames[ii].item())
        batch_mask[ii, :frames] = 1.

    # 将数据放入GPU中
    if use_cuda:
        batch_dev_data = batch_dev_data.to(device)
        batch_mask = batch_mask.to(device)
        batch_target = batch_target.to(device)

    with torch.no_grad():
        acc, loss = train_module(batch_dev_data, batch_mask, batch_target)

    loss = loss.sum() / step_batch_size

    toc = time.time()
    step_time = toc - tic

    dev_loss += float(loss.item())
    dev_acc += float(acc)
    dev_batch_num += 1
    torch.cuda.empty_cache()

epoch_toc = time.time()
epoch_time = epoch_toc - epoch_tic
acc = dev_acc / dev_batch_num
logging.info('Epoch:%d, dev-acc:%.6f, dev-loss:%.6f, cost time :%.6fs', epoch, acc, dev_loss / dev_batch_num,
             epoch_time)
print('learning_rate={}'.format(learning_rate))
