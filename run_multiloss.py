import os
import shutil

import matplotlib
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm, trange

from matplotlib import pyplot as plt

from clr import CyclicLR
import csv
import torchvision

matplotlib.use('Agg')


def train(model, verified_points, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval,
          scheduler):
    model.train()
    errors = 0
    age_loss = 0
    theta_loss = 0
    losses = 0
    verified_points = verified_points.to(device=device, dtype=dtype)
    # correct1, correct5 = 0, 0

    for batch_idx, (data, target, paths) in enumerate(tqdm(loader)):
        if isinstance(scheduler, CyclicLR):
            scheduler.batch_step()
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        theta = []
        for i, path in enumerate(paths):
            csv_path = path[:-3] + 'csv'
            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                row1 = next(reader)  # gets the first line
                row1 = [float(i) for i in row1]

                # row1[2] = (row1[2] + 812) / 913
                # row1[5] = (row1[5] + 916) / 1017
                theta.append(row1)

        theta = torch.FloatTensor(theta)
        # target = torch.FloatTensor(target)
        target_theta = theta.to(device=device)

        #        for param_group in optimizer.param_groups:
        #            print(param_group['lr'])

        optimizer.zero_grad()
        pred_theta, output = model(data)
        # pred_theta = torch.tensor(pred_theta).to(device=device, dtype=dtype)
        assert (theta.size(0) == pred_theta.size(
            0)), "The batch sizes of the input images must be same as the generated grid."
        # print('prediction:'+str(pred_stn.size())+';traget_theta'+str(theta.size()))

        # t_pts: target transformed coordinates by dlib
        # p_pts: predicted transformed coordinates by net
        t_pts, p_pts = pts_trans(pred_theta, target_theta, verified_points)


        pts_loss = my_pw_loss(p_pts, t_pts)

        # age_loss = criterion(output, target)  # sum up batch loss
        loss = pts_loss + 0 * age_loss
        losses += pts_loss.item()
        # losses = pts_loss.item() + 0 * age_loss.item()
        # print (theta_loss)

        # loss = criterion(output, target)
        # losses += loss.item()
        loss.backward()
        optimizer.step()

        # corr = correct(output, target, topk=(1, 5))
        # correct1 += corr[0]
        # correct5 += corr[1]
        cu_mae = f_mae(output, target)
        errors += cu_mae

        if batch_idx % log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'MAE: {:.2f}({:.2f}). '.format(epoch, batch_idx, len(loader),
                                               100. * batch_idx / len(loader), loss.item(),
                                               cu_mae,
                                               errors / (batch_idx + 1)))
        if batch_idx >= 1200:
            break

        if batch_idx == 20:
            p_theta, transformed_input_tensor = model.module.stnmod(data)
            p_theta, transformed_input_tensor = p_theta.cpu(), transformed_input_tensor.cpu()
            out_grid = convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor))
            f, axarr = plt.subplots(figsize=(20, 10))
            axarr.imshow(out_grid)
            axarr.set_title('Transformed Images')
            fig_path = 'results/gif_224_20/'
            if not os.path.exists(fig_path):
                os.mkdir(fig_path)
            f.savefig(fig_path + 'train_epoch_' + str(epoch) + '.png')
            with open('results/224_20_train-batch20.csv', 'a') as csv_file:
                wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
                wr.writerow(p_theta)
                wr.writerow(target_theta.cpu())

    return losses / (batch_idx + 1), errors / (batch_idx + 1)


def test(model, verified_points, epoch, loader, criterion, device, dtype):
    model.eval()
    test_loss = 0
    pts_loss = 0
    age_loss = 0
    theta_loss = 0
    # correct1, correct5 = 0, 0
    errors = 0
    verified_points = verified_points.to(device=device, dtype=dtype)

    for batch_idx, (data, target, paths) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        theta = []
        for i, path in enumerate(paths):
            csv_path = path[:-3] + 'csv'
            with open(csv_path, newline='') as f:
                reader = csv.reader(f)
                row1 = next(reader)  # gets the first line
                row1 = [float(i) for i in row1]
                theta.append(row1)

        theta = torch.FloatTensor(theta)
        theta = theta.to(device=device)

        with torch.no_grad():
            pred_theta, output = model(data)
            assert (theta.size(0) == pred_theta.size(
                0)), "The batch sizes of the input images must be same as the generated grid."
            # print('prediction:'+str(pred_stn.size())+';traget_theta'+str(theta.size()))
            t_pts, p_pts = pts_trans(pred_theta, theta, verified_points)
            pts_loss += my_pw_loss(p_pts, t_pts).item()
            # age_loss += criterion(output, target).item()  # sum up batch loss
            corr = correct(output, target, topk=(1, 5))
            # correct1 += corr[0]
            # correct5 += corr[1]
            cu_mae = f_mae(output, target)
            errors += cu_mae

            if batch_idx == 20:
                p_theta, transformed_input_tensor = model.module.stnmod(data)
                p_theta, transformed_input_tensor = p_theta.cpu(), transformed_input_tensor.cpu()
                out_grid = convert_image_np(
                    torchvision.utils.make_grid(transformed_input_tensor))
                f, axarr = plt.subplots(figsize=(20, 10))
                axarr.imshow(out_grid)
                axarr.set_title('Transformed Images')
                fig_path = 'results/gif_test_224_20/'
                if not os.path.exists(fig_path):
                    os.mkdir(fig_path)
                f.savefig(fig_path + 'epoch_' + str(epoch) + '.png')
                with open('results/test-batch20.csv', 'a') as csv_file:
                    wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
                    wr.writerow(p_theta)

    test_loss = pts_loss
    test_loss /= len(loader)
    tqdm.write(
        '\nTest set: Average loss: {:.4f}.'
        'MAE: {:.2f}.'.format(test_loss, errors / (batch_idx + 1)))
    return test_loss, errors / (batch_idx + 1)


def f_mae(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        values, indices = torch.max(output, 1)
        mae_error = torch.sum(torch.abs(indices - target)) / batch_size

        return mae_error.item()


def correct(output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res


def save_checkpoint(state, is_best, filepath='./result', filename='checkpoint.pth.tar'):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)


def find_bounds_clr(model, loader, optimizer, criterion, device, dtype, min_lr=8e-6, max_lr=8e-5, step_size=2000,
                    mode='triangular', save_path='.'):
    model.train()
    correct1, correct5 = 0, 0
    scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size=step_size, mode=mode)
    epoch_count = step_size // len(loader)  # Assuming step_size is multiple of batch per epoch
    accuracy = []
    for _ in trange(epoch_count):
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            if scheduler is not None:
                scheduler.batch_step()
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)

            optimizer.zero_grad()
            pred_stn, output = model(data)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            corr = correct(output, target)
            accuracy.append(corr[0] / data.shape[0])

    lrs = np.linspace(min_lr, max_lr, step_size)
    plt.plot(lrs, accuracy)
    plt.show()
    plt.savefig(os.path.join(save_path, 'find_bounds_clr.png'))
    np.save(os.path.join(save_path, 'acc.npy'), accuracy)
    return


# Calculating the transformed predefined coordinates location
def pts_trans(predict, target, pts):
    batch_size = target.size(0)
    # add a batch wise dimension and copy the data to every batch
    pts = pts.unsqueeze(0).repeat(batch_size, 1, 1)
    # reconstruct the transformer matrix
    predict = predict.view(batch_size, 2, 3)
    target = target.view(batch_size, 2, 3)
    # get the location of predefined coordinate on target and predict picture
    pts_target = torch.bmm(pts, torch.transpose(target, 1, 2))
    pts_predict = torch.bmm(pts, torch.transpose(predict, 1, 2))

    return pts_target, pts_predict

def pw_loss(p_pts,t_pts):
    p_pts = torch.transpose(p_pts, 1, 2)
    t_pts = torch.transpose(t_pts, 1, 2)
    pw = torch.nn.PairwiseDistance()
    cs = torch.nn.CosineSimilarity()
    pw_loss = pw(p_pts, t_pts)
    pw_loss = torch.mean(pw_loss)
    return pw_loss

def my_pw_loss(p_pts,t_pts):
    # batch_size = t_pts.size(0)
    distance = abs(p_pts - t_pts)
    distance = distance.pow(2)
    distance = torch.sum(distance, dim=2)
    distance = distance ** (1 / 2)
    distance = torch.sum(distance, dim=1)
    distance = torch.mean(distance)
    return distance*1

def my_loss(p_pts,t_pts):

    return sum(sum(sum(abs(p_pts-t_pts))))/(t_pts.size(0)*225)


def convert_image_np(inp):
    with torch.no_grad():
        """Convert a Tensor to numpy image."""
        inp = inp.detach().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp

