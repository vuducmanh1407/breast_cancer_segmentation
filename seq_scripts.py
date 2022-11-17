import os
import pdb
import sys
import copy
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.metrics import calculate_metrics


def seq_train(loader, model, optimizer, device, epoch_idx, recoder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    for batch_idx, data in enumerate(loader):
        image = device.data_to_device(data[0])
        mask = device.data_to_device(data[1])        
        logit = model(image)
        loss = model.loss_calculation(logit, mask)
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data[-1])
            continue
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        optimizer.step()
        loss_value.append(loss.item())
        if batch_idx % recoder.log_interval == 0:
            recoder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
    optimizer.scheduler.step()
    recoder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value


def seq_eval(cfg, loader, model, device, mode, epoch, work_dir, recoder,
             evaluate_tool="python"):
    model.eval()

    metrics = 0
    numOfData = 0
    for batch_idx, data in enumerate(tqdm(loader)):
        recoder.record_timer("device")
        image = device.data_to_device(data[0])
        mask = device.data_to_device(data[1]) 
        with torch.no_grad():
            logit = model(image)
        
        numOfData += image.size(0)
        metrics += calculate_metrics(nn.Sigmoid()(logit), mask, type="mae")*image.size(0)
    try:
        ret = metrics/numOfData
    except:
        print("Unexpected error:", sys.exc_info()[0])
        ret = 100.0
    finally:
        pass
    recoder.print_log(f"Epoch {epoch}, {mode} {ret: .4f}", f"{work_dir}/{mode}.txt")
    return ret


def seq_feature_generation(loader, model, device, mode, work_dir, recoder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
    	os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recoder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))
