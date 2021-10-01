import os
import json
import argparse
import time
from datetime import datetime

import cv2
import torch.nn as nn
from torch.utils.data import Dataset, IterableDataset, DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from spatial_utils import *
from data_utils import *
from gan_utils import * 
from models import *

def train(args):
    test = args.test
    dname = args.dname
    if dname == "turbulent_flows":
        time_steps = 7
    else:
        time_steps = args.time_steps
    batch_size = args.batch_size
    path = args.path
    seed = args.seed
    save_freq = args.save_freq

    if dname == 'lgcp':
        dataset, x_height, x_width = fetch_lgcp(time_steps=time_steps)
    elif dname == 'extreme_weather':
        dataset, x_height, x_width = fetch_extreme_weather(time_steps=time_steps)
    elif dname == 'turbulent_flows':
        dataset, x_height, x_width = fetch_turbulent_flows()

    # Calculate spatio-temporal embedding
    embedding_op = args.embedding_op
    stx_method = args.stx_method
    b = args.dec_weight
    #b = torch.exp(-torch.arange(1, time_steps).flip(0).float() / b).view(1, 1, -1)
    b = temporal_weights(time_steps,b)
    if stx_method=="kw":
      b = 1 / -torch.log(b[0,0,0]) * time_steps-1 # Get temporal weight b from computed weight tensor of length n
      b = torch.exp(-torch.stack([torch.abs(torch.arange(0, time_steps) - t) for t in range(0,time_steps)]) / b)
    
    
    w_sparse = make_sparse_weight_matrix(x_height, x_width)
    if args.embedding_op == "moran":
        data_emb = make_mis(dataset.data, w_sparse)
    elif args.embedding_op == "bea":
        data_emb = make_bis(dataset.data, w_sparse, b, stx_method)
    else:
        data_emb = dataset.data
    # Concatenate data
    data = torch.cat((dataset.data, data_emb), dim=2)
    dataset_full = MyDataset(data)
    scale = args.scale
    para_lam = args.lam

    # filter size for (de)convolutional layers
    g_state_size = args.g_state_size
    d_state_size = args.d_state_size
    g_filter_size = args.g_filter_size
    d_filter_size = args.d_filter_size
    reg_penalty = args.reg_penalty
    nlstm = args.n_lstm
    channels = args.n_channels
    bn = args.batch_norm
    # Number of RNN layers stacked together
    gen_lr = args.lr
    disc_lr = args.lr
    np.random.seed(seed)

    it_counts = 0
    sinkhorn_eps = args.sinkhorn_eps
    sinkhorn_l = args.sinkhorn_l
    # scaling_coef = 1.0
    loader = DataLoader(dataset_full, batch_size=batch_size * 2, drop_last=True)
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # Create instances of generator, discriminator_h and
    # discriminator_m CONV VERSION
    z_width = args.z_dims_t
    z_height = args.z_dims_t
    y_dim = args.y_dims
    j_dims = 16

    generator = VideoDCG(batch_size, time_steps, x_h=x_height, x_w=x_width, filter_size=g_filter_size,
                         state_size=g_state_size, bn=bn, output_act='sigmoid', nchannel=channels).to(device)
    if args.embedding_op == "none":
        discriminator_h = VideoDCD(batch_size, x_h=x_height, x_w=x_width, filter_size=d_filter_size, j=j_dims,
                                   nchannel=channels, bn=bn).to(device)
        discriminator_m = VideoDCD(batch_size, x_h=x_height, x_w=x_width, filter_size=d_filter_size, j=j_dims,
                                   nchannel=channels, bn=bn).to(device)
    else:
        discriminator_h = VideoDCD(batch_size, x_h=x_height, x_w=x_width, filter_size=d_filter_size, j=j_dims,
                                   nchannel=channels*2, bn=bn).to(device)
        discriminator_m = VideoDCD(batch_size, x_h=x_height, x_w=x_width, filter_size=d_filter_size, j=j_dims,
                                   nchannel=channels*2, bn=bn).to(device)

    test_ = dname + "-" + args.loss_func + '-' + args.embedding_op
    
    if args.embedding_op=="bea":
      test_ = test_ + '-' + args.stx_method
    if args.stx_method=="ow":
      test_ = test_ + "l" + str(args.dec_weight)
      
    saved_file = "{}_{}{}-{}:{}:{}.{}".format(test_,
                                              datetime.now().strftime("%h"),
                                              datetime.now().strftime("%d"),
                                              datetime.now().strftime("%H"),
                                              datetime.now().strftime("%M"),
                                              datetime.now().strftime("%S"),
                                              datetime.now().strftime("%f"))

    log_dir = "./trained/{}/log".format(saved_file)

    # Create directories for storing images later.
    if not os.path.exists("trained/{}/data".format(saved_file)):
        os.makedirs("trained/{}/data".format(saved_file))
    if not os.path.exists("trained/{}/images".format(saved_file)):
        os.makedirs("trained/{}/images".format(saved_file))

    # GAN train notes
    with open("./trained/{}/train_notes.txt".format(saved_file), 'w') as f:
        # Include any experiment notes here:
        f.write("Experiment notes: .... \n\n")
        f.write("MODEL_DATA: {}\nSEQ_LEN: {}\n".format(
            test_,
            time_steps))
        f.write("STATE_SIZE: {}\nLAMBDA: {}\n".format(
            g_state_size,
            reg_penalty))
        f.write("BATCH_SIZE: {}\nCRITIC_ITERS: {}\nGenerator LR: {}\n".format(
            batch_size,
            gen_lr,
            disc_lr))
        f.write("SINKHORN EPS: {}\nSINKHORN L: {}\n\n".format(
            sinkhorn_eps,
            sinkhorn_l))

    writer = SummaryWriter(log_dir)

    beta1 = 0.5
    beta2 = 0.9
    optimizerG = optim.Adam(generator.parameters(), lr=gen_lr, betas=(beta1, beta2))
    optimizerDH = optim.Adam(discriminator_h.parameters(), lr=disc_lr, betas=(beta1, beta2))
    optimizerDM = optim.Adam(discriminator_m.parameters(), lr=disc_lr, betas=(beta1, beta2))

    epochs = args.n_epochs
    #loss_lst = []

    w_sparse = w_sparse.to(device)
    b = b.to(device)

    for e in range(epochs):
        for x in loader:
            it_counts += 1
            # Train D
            x1 = x[:, :, :(channels//2)+1, :, :].reshape(batch_size * 2, time_steps, channels, x_height, x_width).to(device)
            if args.stx_method == "skw":
              x2 = x[:, 1:, (channels//2)+1: , :, :].reshape(batch_size * 2, time_steps - 1, channels, x_height, x_width).to(device)
            else:
              x2 = x[:, :, (channels//2)+1: , :, :].reshape(batch_size * 2, time_steps, channels, x_height, x_width).to(device)
            z = torch.randn(batch_size, time_steps, z_height * z_width).to(device)
            y = torch.randn(batch_size, y_dim).to(device)
            z_p = torch.randn(batch_size, time_steps, z_height * z_width).to(device)
            y_p = torch.randn(batch_size, y_dim).to(device)
            real_data = x1[:batch_size, ...]
            real_data_p = x1[batch_size:, ...]
            real_data_emb = x2[:batch_size, ...]
            real_data_p_emb = x2[batch_size:, ...]

            fake_data = generator(z, y).reshape(batch_size, time_steps, channels, x_height, x_width)
            fake_data_p = generator(z_p, y_p).reshape(batch_size, time_steps, channels, x_height, x_width)

            if args.embedding_op == "moran":
                fake_data_emb = make_mis(fake_data, w_sparse)#[:, 1:, :, :, :]
                fake_data_p_emb = make_mis(fake_data_p, w_sparse)#[:, 1:, :, :, :]
            elif args.embedding_op == "spate":
                if args.stx_method == "skw":
                  fake_data_emb = make_spates(fake_data, w_sparse, b, stx_method)[:, 1:, :, :, :]
                  fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method)[:, 1:, :, :, :]
                else:
                  fake_data_emb = make_spates(fake_data, w_sparse, b, stx_method)#[:, 1:, :, :, :]
                  fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method)#[:, 1:, :, :, :]
            else:
                fake_data_emb = None
                fake_data_p_emb = None

            if fake_data_emb is not None:
                if args.stx_method == "skw":
                  real_emb = torch.cat((torch.unsqueeze(real_data[:, 0, :, :, :], 1), real_data_emb), 1)
                  fake_emb = torch.cat((torch.unsqueeze(fake_data[:, 0, :, :, :], 1), fake_data_emb), 1)
                else:
                  real_emb = real_data_emb
                  fake_emb = fake_data_emb
                concat_real = torch.cat((real_data, real_emb), dim=2)
                concat_fake = torch.cat((fake_data, fake_emb), dim=2)

                if args.loss_func == "sinkhorngan":
                    concat_real = concat_real.reshape(batch_size, time_steps, -1)
                    concat_fake = concat_fake.reshape(batch_size, time_steps, -1)
                    loss_d = original_sinkhorn_loss(concat_real, concat_fake, sinkhorn_eps, sinkhorn_l, scale=scale)
                    disc_loss = -loss_d
                else:
                    if args.stx_method == "skw":
                        real_emb_p = torch.cat((torch.unsqueeze(real_data_p[:, 0, :, :, :], 1), real_data_p_emb), 1)
                        fake_emb_p = torch.cat((torch.unsqueeze(fake_data_p[:, 0, :, :, :], 1), fake_data_p_emb), 1)
                    else:
                        real_emb_p = real_data_p_emb
                        fake_emb_p = fake_data_p_emb
                    #return real_data_p, real_emb_p
                    concat_real_p = torch.cat((real_data_p, real_emb_p), dim=2)
                    concat_fake_p = torch.cat((fake_data_p, fake_emb_p), dim=2)

                    # second returned output isn't used.
                    h_fake, h_fake_emb = discriminator_h(concat_fake, concat_fake_p)
                    m_real, m_real_emb = discriminator_m(concat_real, concat_real_p)
                    m_fake, m_fake_emb = discriminator_m(concat_fake, concat_fake_p)
                    h_real_p, h_real_p_emb = discriminator_h(concat_real_p, concat_real)
                    h_fake_p, h_fake_p_emb = discriminator_h(concat_fake_p, concat_fake)
                    m_real_p, m_real_p_emb = discriminator_m(concat_real_p, concat_real)

                    loss_d = compute_mixed_sinkhorn_loss(concat_real, concat_fake, m_real, m_fake, h_fake,
                                                         sinkhorn_eps, sinkhorn_l, concat_real_p, concat_fake_p,
                                                         m_real_p, h_real_p, h_fake_p, scale=scale)
                    pm1 = scale_invariante_martingale_regularization(m_real, reg_penalty, scale=scale)
                    pm2 = scale_invariante_martingale_regularization(m_real_emb, reg_penalty, scale=scale)
                    disc_loss = -loss_d + pm1 + pm2
            else:
                if args.loss_func == "sinkhorngan":
                    real_data = real_data.reshape(batch_size, time_steps, -1)
                    fake_data = fake_data.reshape(batch_size, time_steps, -1)

                    loss_d = original_sinkhorn_loss(real_data, fake_data, sinkhorn_eps, sinkhorn_l, scale=scale)
                    disc_loss = -loss_d
                else:
                    h_fake = discriminator_h(fake_data)

                    m_real = discriminator_m(real_data)
                    m_fake = discriminator_m(fake_data)

                    h_real_p = discriminator_h(real_data_p)
                    h_fake_p = discriminator_h(fake_data_p)

                    m_real_p = discriminator_m(real_data_p)

                    real_data = real_data.reshape(batch_size, time_steps, -1)
                    fake_data = fake_data.reshape(batch_size, time_steps, -1)
                    real_data_p = real_data_p.reshape(batch_size, time_steps, -1)
                    fake_data_p = fake_data_p.reshape(batch_size, time_steps, -1)

                    loss_d = compute_mixed_sinkhorn_loss(real_data, fake_data, m_real, m_fake, h_fake,
                                                         sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p,
                                                         m_real_p, h_real_p, h_fake_p, scale=scale)

                    pm1 = scale_invariante_martingale_regularization(m_real, reg_penalty, scale=scale)
                    disc_loss = -loss_d + pm1
            # torch.autograd.set_detect_anomaly(True)

            # updating Discriminators
            discriminator_h.zero_grad()
            discriminator_m.zero_grad()
            disc_loss.backward()
            optimizerDH.step()
            optimizerDM.step()

            # Train G
            z = torch.randn(batch_size, time_steps, z_height * z_width).to(device)
            y = torch.randn(batch_size, y_dim).to(device)
            z_p = torch.randn(batch_size, time_steps, z_height * z_width).to(device)
            y_p = torch.randn(batch_size, y_dim).to(device)

            fake_data = generator(z, y).reshape(batch_size, time_steps, channels, x_height, x_width)
            fake_data_p = generator(z_p, y_p).reshape(batch_size, time_steps, channels, x_height, x_width)

            if args.embedding_op == "moran":
                fake_data_emb = make_mis(fake_data, w_sparse)#[:, 1:, :, :, :]
                fake_data_p_emb = make_mis(fake_data_p, w_sparse)#[:, 1:, :, :, :]
            elif args.embedding_op == "spate":
                if args.stx_method == "skw":
                  fake_data_emb = make_spates(fake_data, w_sparse, b, stx_method)[:, 1:, :, :, :]
                  fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method)[:, 1:, :, :, :]
                else:
                  fake_data_emb = make_spates(fake_data, w_sparse, b, stx_method)#[:, 1:, :, :, :]
                  fake_data_p_emb = make_spates(fake_data_p, w_sparse, b, stx_method)#[:, 1:, :, :, :]
            else:
                fake_data_emb = None
                fake_data_p_emb = None

            if fake_data_emb is not None:
                if args.stx_method == "skw":
                    real_emb = torch.cat((torch.unsqueeze(real_data[:, 0, :, :, :], 1), real_data_emb), 1)
                    fake_emb = torch.cat((torch.unsqueeze(fake_data[:, 0, :, :, :], 1), fake_data_emb), 1)
                else:
                    real_emb = real_data_emb
                    fake_emb = fake_data_emb
                concat_real = torch.cat((real_data, real_emb), dim=2)
                concat_fake = torch.cat((fake_data, fake_emb), dim=2)

                if args.loss_func == "sinkhorngan":
                    concat_real = concat_real.reshape(batch_size, time_steps, -1)
                    concat_fake = concat_fake.reshape(batch_size, time_steps, -1)
                    loss_g = original_sinkhorn_loss(concat_real, concat_fake, sinkhorn_eps, sinkhorn_l, scale=scale)
                else:
                    if args.stx_method == "skw":
                        real_emb_p = torch.cat((torch.unsqueeze(real_data_p[:, 0, :, :, :], 1), real_data_p_emb), 1)
                        fake_emb_p = torch.cat((torch.unsqueeze(fake_data_p[:, 0, :, :, :], 1), fake_data_p_emb), 1)
                    else:
                        real_emb_p = real_data_p_emb
                        fake_emb_p = fake_data_p_emb
                    concat_real_p = torch.cat((real_data_p, real_emb_p), dim=2)
                    concat_fake_p = torch.cat((fake_data_p, fake_emb_p), dim=2)

                    # second returned output isn't used.
                    h_fake, h_fake_emb = discriminator_h(concat_fake, concat_fake_p)
                    m_real, m_real_emb = discriminator_m(concat_real, concat_real_p)
                    m_fake, m_fake_emb = discriminator_m(concat_fake, concat_fake_p)
                    h_real_p, h_real_p_emb = discriminator_h(concat_real_p, concat_real)
                    h_fake_p, h_fake_p_emb = discriminator_h(concat_fake_p, concat_fake)
                    m_real_p, m_real_p_emb = discriminator_m(concat_real_p, concat_real)

                    loss_g = compute_mixed_sinkhorn_loss(concat_real, concat_fake, m_real, m_fake, h_fake,
                                                         sinkhorn_eps, sinkhorn_l, concat_real_p, concat_fake_p,
                                                         m_real_p, h_real_p, h_fake_p, scale=scale)
            else:
                if args.loss_func == "sinkhorngan":
                    real_data = real_data.reshape(batch_size, time_steps, -1)
                    fake_data = fake_data.reshape(batch_size, time_steps, -1)
                    loss_g = original_sinkhorn_loss(real_data, fake_data, sinkhorn_eps, sinkhorn_l, scale=scale)
                else:
                    h_fake = discriminator_h(fake_data)

                    m_real = discriminator_m(real_data)
                    m_fake = discriminator_m(fake_data)

                    h_real_p = discriminator_h(real_data_p)
                    h_fake_p = discriminator_h(fake_data_p)

                    m_real_p = discriminator_m(real_data_p)

                    real_data = real_data.reshape(batch_size, time_steps, -1)
                    fake_data = fake_data.reshape(batch_size, time_steps, -1)
                    real_data_p = real_data_p.reshape(batch_size, time_steps, -1)
                    fake_data_p = fake_data_p.reshape(batch_size, time_steps, -1)

                    loss_g = compute_mixed_sinkhorn_loss(real_data, fake_data, m_real, m_fake, h_fake,
                                                         sinkhorn_eps, sinkhorn_l, real_data_p, fake_data_p,
                                                         m_real_p, h_real_p, h_fake_p, scale=scale)
            gen_loss = loss_g
            #loss_lst.append(gen_loss)
            # updating Generator
            generator.zero_grad()
            gen_loss.backward()
            optimizerG.step()
            # it.set_postfix(loss=float(gen_loss))
            # it.update(1)

            # ...log the running loss
            writer.add_scalar('Sinkhorn training loss', gen_loss, it_counts)
            if args.loss_func == "cotgan":
                writer.add_scalar('pM for real', pm1, it_counts)
                if not args.embedding_op == "none":
                    writer.add_scalar('pM for embedding', pm2, it_counts)
                writer.flush()

            if torch.isinf(gen_loss):
                print('%s Loss exploded!' % test_)
                # Open the existing file with mode a - append
                with open("./trained/{}/train_notes.txt".format(saved_file), 'a') as f:
                    # Include any experiment notes here:
                    f.write("\n Training failed! ")
                break
            else:
                if it_counts % save_freq == 0 or it_counts == 1:
                    print("Epoch [%d/%d] - Generator Loss: %f - Discriminator Loss: %f" % (
                    e, epochs, gen_loss.item(), disc_loss.item()))
                    z = torch.randn(batch_size, time_steps, z_height * z_width).to(device)
                    y = torch.randn(batch_size, y_dim).to(device)
                    samples = generator(z, y)
                    # plot first 5 samples within one image
                    '''
                    plot1 = torch.squeeze(samples[0]).permute(1, 0, 2)
                    plt.figure()
                    plt.imshow(plot1.reshape([x_height, 10 * x_width]).detach().numpy())
                    plt.show()
                    '''
                    # print(samples.shape)
                    n_show = min(batch_size, 5)
                    samples = samples[:n_show, :, 0, :, :].permute(0, 2, 1, 3)
                    img = samples.reshape(1, n_show * x_height, time_steps * x_width)
                    writer.add_image('Generated images', img, global_step=it_counts)
                    #  save model to file
                    save_path = "./trained/{}/ckpts".format(saved_file)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(generator, save_path + '/' + 'generator.pt') #Only save final generator
                    #torch.save(generator, save_path + '/' + 'generator{}.pt'.format(it_counts)) #Save steps
                    #torch.save(discriminator_h, save_path + '/' + 'discriminatorH{}.pt'.format(it_counts))
                    #torch.save(discriminator_m, save_path + '/' + 'discriminatorM{}.pt'.format(it_counts))
                    print("Saved all models to {}".format(save_path))
            continue
    writer.close()



if __name__ == '__main__':
          parser = argparse.ArgumentParser(description='cot')
          parser.add_argument('-d', '--dname', type=str, default="lgcp",
                              choices=['lgcp', 'extreme_weather', 'turbulent_flows'])
          parser.add_argument('-lf', '--loss_func', type=str, default="cotgan", choices=["sinkhorngan", "cotgan"])
          parser.add_argument('-eo', '--embedding_op', type=str, default="spate", choices=["moran", "spate", "none"])
          parser.add_argument('-stx', '--stx_method', type=str, default="skw", choices=["skw", "k", "kw"])
          parser.add_argument('-t', '--test', type=str, default='cot', choices=['cot'])
          parser.add_argument('-s', '--seed', type=int, default=1)
          parser.add_argument('-b', '--dec_weight', type=int, default=20)
          parser.add_argument('-gss', '--g_state_size', type=int, default=32)
          parser.add_argument('-gfs', '--g_filter_size', type=int, default=32)
          parser.add_argument('-dss', '--d_state_size', type=int, default=32)
          parser.add_argument('-dfs', '--d_filter_size', type=int, default=32)
          parser.add_argument('-ts', '--time_steps', type=int, default=10)
          parser.add_argument('-sinke', '--sinkhorn_eps', type=float, default=0.8)
          parser.add_argument('-reg_p', '--reg_penalty', type=float, default=1.5)
          parser.add_argument('-sinkl', '--sinkhorn_l', type=int, default=100)
          parser.add_argument('-Dx', '--Dx', type=int, default=1)
          parser.add_argument('-Dz', '--z_dims_t', type=int, default=5)
          parser.add_argument('-Dy', '--y_dims', type=int, default=20)
          parser.add_argument('-g', '--gen', type=str, default="fc", choices=["lstm", "fc"])
          parser.add_argument('-bs', '--batch_size', type=int, default=32)
          parser.add_argument('-p', '--path', type=str, default='./')
          parser.add_argument('-save', '--save_freq', type=int, default=5)
          parser.add_argument('-ne', '--n_epochs', type=int, default=30)
          parser.add_argument('-lr', '--lr', type=float, default=1e-4)
          parser.add_argument('-bn', '--batch_norm', type=bool, default=True)
          parser.add_argument('-sl', '--scale', type=bool, default=True)
          parser.add_argument('-nlstm', '--n_lstm', type=int, default=1)
          parser.add_argument('-lam', '--lam', type=float, default=1.0)
      
          parser.add_argument('-nch', '--n_channels', type=int, default=1)
          parser.add_argument('-rt', '--read_tfrecord', type=bool, default=True)
          parser.add_argument('-f')  # Dummy to get parser to run in Colab
      
          args = parser.parse_args()
          print("TRAINING - Dataset: " + args.dname + " Emb: " + args.embedding_op + " Loss: " + args.loss_func)
          train(args)