import os
import math
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from loggers.logger import LOG
from core.utils import *
from core.builder.build_convrnn import ConvRNN
from data_provider.data_iterator import CIKM_sample, Shanghai_sample
from default_configs.default_config import DefaultConfig
from model_values.evaluation_functions import *

LOG = LOG.get_instance().get_logger()

class CIKM_ModelsTrainer:
    def __init__(self, architecture, generation):
        self.architecture = architecture
        self.generation = generation
        self.convrnn = ConvRNN(architecture, 'CIKM').cuda()
        self.optimizer = Adam(self.convrnn.parameters(), lr=DefaultConfig.get_config('lr'))
        self.MSE_criterion = nn.MSELoss(reduction='mean')
        self.MAE_criterion = nn.L1Loss(reduction='mean')
        self.test_MSE_criterion = MSE()
        self.test_MAE_criterion = MAE()
        self.SSIM = SSIM(window_size=5,img_channel=DefaultConfig.get_config('CIKM_img_channel'))
        self.CSI = CSI()
        self.HSS = HSS()
        self.Precision = Precision()
        self.Accuracy = Accuracy()
        self.Recall = Recall()
        self.F1_score = F1_score()

    def save(self, train_epoch=None):
        if not os.path.exists(os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), 'CIKM')):
            os.makedirs(os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), 'CIKM'))
        checkpoint_path = os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), 'CIKM', f'{self.architecture.identifier}.ckpt')
        stats = {}
        stats['net_param'] = self.convrnn.state_dict()
        stats['train_epoch'] = train_epoch
        torch.save(stats, checkpoint_path)
        LOG.info("Save the model parameters to %s" % checkpoint_path)

    def load(self):
        checkpoint_path = os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), 'CIKM', f'{self.architecture.identifier}.ckpt')
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f'There are no trained model parameters under the {checkpoint_path} directory.')
        stats = torch.load(checkpoint_path, weights_only=False)
        self.convrnn.load_state_dict(stats['net_param'])
        LOG.info(f'The parameters of the model {self.architecture.identifier} have been loaded.')
        return stats

    def weight_sharing(self):
        parameter = {}
        for parent in self.architecture.parents.keys():
            checkpoint_path = os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), f'{parent}.ckpt')
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f'There are no trained model parameters under the {checkpoint_path} directory.')
            stats = torch.load(checkpoint_path, weights_only=False)
            for name in stats['net_param'].keys():
                for layer_id in self.architecture.parents[parent]:
                    if f'cell_list.{layer_id}' in name:
                        if name in self.convrnn.state_dict().keys():
                            if stats['net_param'][name].shape == self.convrnn.state_dict()[name].shape:
                                parameter[name] = stats['net_param'][name]
        return parameter

    def train(self, model_all_values):
        if os.path.exists(os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), f'{self.architecture.identifier}.ckpt')):
            stats = self.load()
        else:
            if len(self.architecture.parents) == 2:
                parameter = self.weight_sharing()
                self.convrnn.load_state_dict(parameter, strict=False)

        eta = DefaultConfig.get_config('sampling_start_value')
        average_mse_loss = average_mae_loss = 0
        is_reduce_train_epoch = False
        if is_reduce_train_epoch:
            train_epoch = DefaultConfig.get_config('max_iterations') - self.generation * ((DefaultConfig.get_config('max_iterations') - 200) // DefaultConfig.get_config('number_of_generations'))
        else:
            train_epoch = DefaultConfig.get_config('max_iterations')
        start_time = time.time()
        print('-' * 75 + 'begin training' + '-' * 75)
        pbar_train = tqdm(total=train_epoch, ncols=150)
        for itr in range(1, train_epoch + 1):
            ims = CIKM_sample(batch_size=DefaultConfig.get_config('batch_size'))
            ims = padding_CIKM_data(ims)
            ims = reshape_patch(ims, DefaultConfig.get_config('patch_size'))
            ims = nor(ims)
            eta, real_input_flag = schedule_CIKM_sampling(eta, itr)
            frames_tensor = torch.FloatTensor(ims).cuda()
            mask_tensor = torch.FloatTensor(real_input_flag).cuda()
            self.optimizer.zero_grad()
            next_frames = self.convrnn(frames_tensor, mask_tensor)
            MSE_loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
            MAE_loss = self.MAE_criterion(next_frames, frames_tensor[:, 1:])
            loss = MSE_loss + MAE_loss
            loss.backward(retain_graph=True)
            self.optimizer.step()
            mse_loss = MSE_loss.detach().cpu().numpy()
            mae_loss = MAE_loss.detach().cpu().numpy()
            if DefaultConfig.get_config('reverse_input'):
                ims_rev = np.flip(ims, axis=1).copy()
                frames_tensor = torch.FloatTensor(ims_rev).cuda()
                mask_tensor = torch.FloatTensor(real_input_flag).cuda()
                self.optimizer.zero_grad()
                next_frames = self.convrnn(frames_tensor, mask_tensor)
                MSE_loss_rev = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
                MAE_loss_rev = self.MAE_criterion(next_frames, frames_tensor[:, 1:])
                loss = MSE_loss_rev + MAE_loss_rev
                loss.backward()
                self.optimizer.step()
                mse_loss = (mse_loss + MSE_loss_rev.detach().cpu().numpy()) / 2
                mae_loss = (mae_loss + MAE_loss_rev.detach().cpu().numpy()) / 2
            average_mse_loss = average_mse_loss + mse_loss
            average_mae_loss = average_mae_loss + mae_loss
            pbar_train.update(1)
            pbar_train.set_description('Training progress')
            pbar_train.set_postfix({'MSE_loss': '{:.3f}'.format(mse_loss), 'MAE_loss': '{:.3f}'.format(mae_loss), 'model_name': f'{self.architecture.identifier}'})
            if np.isnan(mae_loss) or np.isnan(mse_loss):
                model_all_values.number_of_all_blocks = np.inf
                model_all_values.number_of_parameters = np.inf
                model_all_values.mae_loss = np.inf
                model_all_values.mse_loss = np.inf
                model_all_values.training_time = np.inf
                model_all_values.csi = np.float64(0)
                model_all_values.hss = np.float64(0)
                model_all_values.ssim = np.float64(0)
                model_all_values.accuracy = np.float64(0)
                LOG.info(f'Average training loss of the {self.architecture.identifier} model over {train_epoch} epochs: | [MSE_loss = {average_mse_loss / train_epoch}, MAE_loss = {average_mae_loss / train_epoch}] |')
                return model_all_values
        LOG.info(f'Average training loss of the {self.architecture.identifier} model over {train_epoch} epochs: | [MSE_loss = {average_mse_loss/train_epoch}, MAE_loss = {average_mae_loss/train_epoch}] |')

        model_all_values.training_time = time.time() - start_time
        model_all_values.number_of_parameters = sum(p.numel() for p in self.convrnn.parameters() if p.requires_grad)
        if not os.path.exists(
                os.path.join(DefaultConfig.get_config('test_pred_imgs'), 'CIKM', f'{self.architecture.identifier}')):
            os.mkdir(
                os.path.join(DefaultConfig.get_config('test_pred_imgs'), 'CIKM', f'{self.architecture.identifier}'))
        clean_fold(os.path.join(DefaultConfig.get_config('test_pred_imgs'), 'CIKM', f'{self.architecture.identifier}'))
        mse_loss = mae_loss = ssim = 0
        count = 0
        index = 1
        flag = True
        dBZ_threshold_list = [5, 20, 40]
        indator_dict = {}
        real_input_flag = np.zeros((DefaultConfig.get_config('batch_size'),
                                    DefaultConfig.get_config('total_length') - DefaultConfig.get_config(
                                        'input_length') - 1,
                                    DefaultConfig.get_config('CIKM_img_width') // DefaultConfig.get_config(
                                        'patch_size'),
                                    DefaultConfig.get_config('CIKM_img_height') // DefaultConfig.get_config(
                                        'patch_size'),
                                    DefaultConfig.get_config('patch_size') ** 2 * DefaultConfig.get_config(
                                        'CIKM_img_channel')))
        output_length = DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length')
        LOG.info('-' * 75 + 'begin testing' + '-' * 75)
        while flag:
            dat, (index, b_cup) = CIKM_sample(DefaultConfig.get_config('batch_size'), data_type='test', index=index)
            dat = nor(dat)
            ims = padding_CIKM_data(dat)
            ims = reshape_patch(ims, DefaultConfig.get_config('patch_size'))
            frames_tensor = torch.FloatTensor(ims).cuda()
            mask_tensor = torch.FloatTensor(real_input_flag).cuda()
            next_frames = self.convrnn(frames_tensor, mask_tensor)
            frames_tensor = reshape_patch_back(frames_tensor, DefaultConfig.get_config('patch_size'))
            next_frames = reshape_patch_back(next_frames, DefaultConfig.get_config('patch_size'))
            frames_tensor = unpadding_CIKM_data(frames_tensor)
            next_frames = unpadding_CIKM_data(next_frames)
            MSE_loss = self.test_MSE_criterion(next_frames[:, 4:], frames_tensor[:, 5:]).detach().cpu().numpy()
            MAE_loss = self.test_MAE_criterion(next_frames[:, 4:], frames_tensor[:, 5:]).detach().cpu().numpy()
            SSIM = self.SSIM(next_frames[:, 4:], frames_tensor[:, 5:]).detach().cpu().numpy()
            avg_csi = avg_hss = avg_pre = avg_acc = avg_rec = avg_f1 = 0
            indicators_dict = {}
            avg_dict = {}
            for dBZ_threshold in dBZ_threshold_list:
                indicators = {}
                CSI = self.CSI(next_frames[:, 4:], frames_tensor[:, 5:], dBZ_threshold)
                indicators['CSI'] = float(CSI)
                avg_csi += float(CSI)
                HSS = self.HSS(next_frames[:, 4:], frames_tensor[:, 5:], dBZ_threshold)
                indicators['HSS'] = float(HSS)
                avg_hss += float(HSS)
                Precision = self.Precision(next_frames[:, 4:], frames_tensor[:, 5:], dBZ_threshold)
                indicators['Precision'] = float(Precision)
                avg_pre += float(Precision)
                Accuracy = self.Accuracy(next_frames[:, 4:], frames_tensor[:, 5:], dBZ_threshold)
                indicators['Accuracy'] = float(Accuracy)
                avg_acc += float(Accuracy)
                Recall = self.Recall(next_frames[:, 4:], frames_tensor[:, 5:], dBZ_threshold)
                indicators['Recall'] = float(Recall)
                avg_rec += float(Recall)
                F1_score = self.F1_score(next_frames[:, 4:], frames_tensor[:, 5:], dBZ_threshold)
                indicators['F1_score'] = float(F1_score)
                avg_f1 += float(F1_score)
                indicators_dict[dBZ_threshold] = indicators
            avg_dict['CSI'] = avg_csi / len(dBZ_threshold_list)
            avg_dict['HSS'] = avg_hss / len(dBZ_threshold_list)
            avg_dict['Precision'] = avg_pre / len(dBZ_threshold_list)
            avg_dict['Accuracy'] = avg_acc / len(dBZ_threshold_list)
            avg_dict['Recall'] = avg_rec / len(dBZ_threshold_list)
            avg_dict['F1_score'] = avg_f1 / len(dBZ_threshold_list)
            indicators_dict['average'] = avg_dict
            img_gen = next_frames.detach().cpu().numpy()
            img_out = img_gen[:, -output_length:]
            count = count + 1
            mse_loss = mse_loss + MSE_loss
            mae_loss = mae_loss + MAE_loss
            ssim = ssim + SSIM
            if count == 1:
                indator_dict = indicators_dict
            else:
                for dBZ_threshold in indicators_dict.keys():
                    for name in indicators_dict[dBZ_threshold].keys():
                        indator_dict[dBZ_threshold][name] += float(indicators_dict[dBZ_threshold][name])
            if count % 25 == 0:
                print(f'test progress:|{(count // 25) * 10}%|' + ' [' + '*' * (count // 25) * 10 + '-' * (
                            100 - (count // 25) * 10) + ']')
            if index == 1001:
                flag = False

        mse_loss = mse_loss / count
        mae_loss = mae_loss / count
        if np.isnan(mse_loss):
            model_all_values.mse_loss = np.inf
        else:
            model_all_values.mse_loss = mse_loss
        if np.isnan(mae_loss):
            model_all_values.mae_loss = np.inf
        else:
            model_all_values.mae_loss = mae_loss
        ssim = ssim / count
        model_all_values.ssim = ssim
        for dBZ_threshold in indator_dict.keys():
            for name in indator_dict[dBZ_threshold].keys():
                indator_dict[dBZ_threshold][name] = round(indator_dict[dBZ_threshold][name] / count, 4)
            if dBZ_threshold == 'average':
                model_all_values.csi = indator_dict[dBZ_threshold]['CSI']
                model_all_values.hss = indator_dict[dBZ_threshold]['HSS']

        LOG.info('-' * 75 + 'End Of Test' + '-' * 75)
        self.save(DefaultConfig.get_config('max_iterations'))
        return model_all_values

class Shanghai_ModelsTrainer:
    def __init__(self, architecture, generation):
        self.architecture = architecture
        self.generation = generation
        self.convrnn = ConvRNN(architecture, 'Shanghai').cuda()
        self.optimizer = Adam(self.convrnn.parameters(), lr=DefaultConfig.get_config('lr'))
        self.MSE_criterion = nn.MSELoss(reduction='mean')
        self.MAE_criterion = nn.L1Loss(reduction='mean')
        self.SSIM = SSIM(window_size=5,img_channel=DefaultConfig.get_config('Shanghai_img_channel'))
        self.CSI = CSI()
        self.HSS = HSS()
        self.Precision = Precision()
        self.Accuracy = Accuracy()
        self.Recall = Recall()
        self.F1_score = F1_score()

    def save(self, train_epoch=None):
        if not os.path.exists(os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), 'Shanghai')):
            os.makedirs(os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), 'Shanghai'))
        checkpoint_path = os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), 'Shanghai', f'{self.architecture.identifier}.ckpt')
        stats = {}
        stats['net_param'] = self.convrnn.state_dict()
        stats['train_epoch'] = train_epoch
        torch.save(stats, checkpoint_path)
        LOG.info("保存模型参数到 %s" % checkpoint_path)

    def load(self):
        checkpoint_path = os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), 'Shanghai', f'{self.architecture.identifier}.ckpt')
        if not os.path.exists(checkpoint_path):
            raise RuntimeError(f'There are no trained model parameters under the {checkpoint_path} directory.')
        stats = torch.load(checkpoint_path, weights_only=False)
        self.convrnn.load_state_dict(stats['net_param'])
        LOG.info(f'The parameters of the model {self.architecture.identifier} have been loaded.')
        return stats

    def weight_sharing(self):
        parameter = {}
        for parent in self.architecture.parents.keys():
            checkpoint_path = os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), f'{parent}.ckpt')
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f'There are no trained model parameters under the {checkpoint_path} directory.')
            stats = torch.load(checkpoint_path, weights_only=False)
            for name in stats['net_param'].keys():
                for layer_id in self.architecture.parents[parent]:
                    if f'cell_list.{layer_id}' in name:
                        if name in self.convrnn.state_dict().keys():
                            if stats['net_param'][name].shape == self.convrnn.state_dict()[name].shape:
                                parameter[name] = stats['net_param'][name]
        return parameter

    def train(self, model_all_values):

        if os.path.exists(os.path.join(DefaultConfig.get_config('models_parameter_save_dir'), f'{self.architecture.identifier}.ckpt')):
            stats = self.load()
        else:
            if len(self.architecture.parents) == 2:
                parameter = self.weight_sharing()
                self.convrnn.load_state_dict(parameter, strict=False)

        eta = DefaultConfig.get_config('sampling_start_value')
        average_mse_loss = average_mae_loss = 0
        is_reduce_train_epoch = False
        if is_reduce_train_epoch:
            train_epoch = DefaultConfig.get_config('max_iterations') - self.generation * ((DefaultConfig.get_config('max_iterations') - 200) // DefaultConfig.get_config('number_of_generations'))
        else:
            train_epoch = DefaultConfig.get_config('max_iterations')
        start_time = time.time()
        print('-' * 75 + 'begin training' + '-' * 75)
        pbar_train = tqdm(total=train_epoch, ncols=150)
        for itr in range(1, train_epoch + 1):
            ims = Shanghai_sample(batch_size=DefaultConfig.get_config('batch_size'))
            ims = padding_Shanghai_data(ims)
            ims = reshape_patch(ims, DefaultConfig.get_config('patch_size'))
            ims = nor(ims)
            eta, real_input_flag = schedule_Shanghai_sampling(eta, itr)
            frames_tensor = torch.FloatTensor(ims).cuda()
            mask_tensor = torch.FloatTensor(real_input_flag).cuda()
            self.optimizer.zero_grad()
            next_frames = self.convrnn(frames_tensor, mask_tensor)
            MSE_loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
            MAE_loss = self.MAE_criterion(next_frames, frames_tensor[:, 1:])
            loss = MSE_loss + MAE_loss
            loss.backward(retain_graph=True)
            self.optimizer.step()
            mse_loss = MSE_loss.detach().cpu().numpy()
            mae_loss = MAE_loss.detach().cpu().numpy()
            if DefaultConfig.get_config('reverse_input'):
                ims_rev = np.flip(ims, axis=1).copy()
                frames_tensor = torch.FloatTensor(ims_rev).cuda()
                mask_tensor = torch.FloatTensor(real_input_flag).cuda()
                self.optimizer.zero_grad()
                next_frames = self.convrnn(frames_tensor, mask_tensor)
                MSE_loss_rev = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
                MAE_loss_rev = self.MAE_criterion(next_frames, frames_tensor[:, 1:])
                loss = MSE_loss_rev + MAE_loss_rev
                loss.backward()
                self.optimizer.step()
                mse_loss = (mse_loss + MSE_loss_rev.detach().cpu().numpy()) / 2
                mae_loss = (mae_loss + MAE_loss_rev.detach().cpu().numpy()) / 2
            average_mse_loss = average_mse_loss + mse_loss
            average_mae_loss = average_mae_loss + mae_loss
            pbar_train.update(1)
            pbar_train.set_description('Training progress')
            pbar_train.set_postfix({'MSE_loss': '{:.3f}'.format(mse_loss), 'MAE_loss': '{:.3f}'.format(mae_loss), 'model_name': f'{self.architecture.identifier}'})
            if np.isnan(mae_loss) or np.isnan(mse_loss):
                model_all_values.number_of_all_blocks = np.inf
                model_all_values.number_of_parameters = np.inf
                model_all_values.mae_loss = np.inf
                model_all_values.mse_loss = np.inf
                model_all_values.training_time = np.inf
                model_all_values.csi = np.float64(0)
                model_all_values.hss = np.float64(0)
                model_all_values.ssim = np.float64(0)
                model_all_values.accuracy = np.float64(0)
                LOG.info(f'Average training loss of the {self.architecture.identifier} model over {train_epoch} epochs: | [MSE_loss = {average_mse_loss / train_epoch}, MAE_loss = {average_mae_loss / train_epoch}] |')
                return model_all_values
        LOG.info(f'Average training loss of the {self.architecture.identifier} model over {train_epoch} epochs: | [MSE_loss = {average_mse_loss/train_epoch}, MAE_loss = {average_mae_loss/train_epoch}] |')

        model_all_values.training_time = time.time() - start_time
        model_all_values.number_of_parameters = sum(p.numel() for p in self.convrnn.parameters() if p.requires_grad)
        mse_loss = mae_loss = ssim = 0
        count = 0
        index = 1
        flag = True
        dBZ_threshold_list = [5, 20, 40]
        indator_dict = {}
        real_input_flag = np.zeros((DefaultConfig.get_config('batch_size'), DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length') - 1,
                                    DefaultConfig.get_config('Shanghai_img_width') // DefaultConfig.get_config('patch_size'),
                                    DefaultConfig.get_config('Shanghai_img_height') // DefaultConfig.get_config('patch_size'),
                                    DefaultConfig.get_config('patch_size') ** 2 * DefaultConfig.get_config('Shanghai_img_channel')))
        output_length = DefaultConfig.get_config('total_length') - DefaultConfig.get_config('input_length')
        LOG.info('-' * 75 + 'begin testing' + '-' * 75)
        while flag:
            dat, (index, b_cup) = Shanghai_sample(DefaultConfig.get_config('batch_size'), data_type='test', index=index)
            dat = nor(dat)
            ims = padding_Shanghai_data(dat)
            ims = reshape_patch(ims, DefaultConfig.get_config('patch_size'))
            frames_tensor = torch.FloatTensor(ims).cuda()
            mask_tensor = torch.FloatTensor(real_input_flag).cuda()
            next_frames = self.convrnn(frames_tensor, mask_tensor)
            frames_tensor = reshape_patch_back(frames_tensor, DefaultConfig.get_config('patch_size'))
            next_frames = reshape_patch_back(next_frames, DefaultConfig.get_config('patch_size'))
            frames_tensor = unpadding_Shanghai_data(frames_tensor)
            next_frames = unpadding_Shanghai_data(next_frames)
            MSE_loss = self.MSE_criterion(next_frames[:, 9:], frames_tensor[:, 10:]).detach().cpu().numpy()
            MAE_loss = self.MAE_criterion(next_frames[:, 9:], frames_tensor[:, 10:]).detach().cpu().numpy()
            SSIM = self.SSIM(next_frames[:, 9:], frames_tensor[:, 10:]).detach().cpu().numpy()
            avg_csi = avg_hss = avg_pre = avg_acc = avg_rec = avg_f1 = 0
            indicators_dict = {}
            avg_dict = {}
            for dBZ_threshold in dBZ_threshold_list:
                indicators = {}
                CSI = self.CSI(next_frames[:, 9:], frames_tensor[:, 10:], dBZ_threshold)
                indicators['CSI'] = float(CSI)
                avg_csi += float(CSI)
                HSS = self.HSS(next_frames[:, 9:], frames_tensor[:, 10:], dBZ_threshold)
                indicators['HSS'] = float(HSS)
                avg_hss += float(HSS)
                Precision = self.Precision(next_frames[:, 9:], frames_tensor[:, 10:], dBZ_threshold)
                indicators['Precision'] = float(Precision)
                avg_pre += float(Precision)
                Accuracy = self.Accuracy(next_frames[:, 9:], frames_tensor[:, 10:], dBZ_threshold)
                indicators['Accuracy'] = float(Accuracy)
                avg_acc += float(Accuracy)
                Recall = self.Recall(next_frames[:, 9:], frames_tensor[:, 10:], dBZ_threshold)
                indicators['Recall'] = float(Recall)
                avg_rec += float(Recall)
                F1_score = self.F1_score(next_frames[:, 9:], frames_tensor[:, 10:], dBZ_threshold)
                indicators['F1_score'] = float(F1_score)
                avg_f1 += float(F1_score)
                indicators_dict[dBZ_threshold] = indicators
            avg_dict['CSI'] = avg_csi / len(dBZ_threshold_list)
            avg_dict['HSS'] = avg_hss / len(dBZ_threshold_list)
            avg_dict['Precision'] = avg_pre / len(dBZ_threshold_list)
            avg_dict['Accuracy'] = avg_acc / len(dBZ_threshold_list)
            avg_dict['Recall'] = avg_rec / len(dBZ_threshold_list)
            avg_dict['F1_score'] = avg_f1 / len(dBZ_threshold_list)
            indicators_dict['average'] = avg_dict
            img_gen = next_frames.detach().cpu().numpy()
            img_out = img_gen[:, -output_length:]
            count = count + 1
            mse_loss = mse_loss + MSE_loss
            mae_loss = mae_loss + MAE_loss
            ssim = ssim + SSIM
            if count == 1:
                indator_dict = indicators_dict
            else:
                for dBZ_threshold in indicators_dict.keys():
                    for name in indicators_dict[dBZ_threshold].keys():
                        indator_dict[dBZ_threshold][name] += float(indicators_dict[dBZ_threshold][name])

            if count % 15 == 0:
                print(f'test progress:|{((count*100)//15)//5}%|' + ' [' + '*' * (((count*100)//15)//5) + '-' * (100-((count*100)//15)//5) + ']')
            if index == 301:
                flag = False

        mse_loss = mse_loss / count
        mae_loss = mae_loss / count
        if np.isnan(mse_loss):
            model_all_values.mse_loss = np.inf
        else:
            model_all_values.mse_loss = mse_loss
        if np.isnan(mae_loss):
            model_all_values.mae_loss = np.inf
        else:
            model_all_values.mae_loss = mae_loss
        ssim = ssim / count
        model_all_values.ssim = ssim
        for dBZ_threshold in indator_dict.keys():
            for name in indator_dict[dBZ_threshold].keys():
                indator_dict[dBZ_threshold][name] = round(indator_dict[dBZ_threshold][name] / count, 4)
            if dBZ_threshold == 'average':
                model_all_values.csi = indator_dict[dBZ_threshold]['CSI']
                model_all_values.hss = indator_dict[dBZ_threshold]['HSS']

        LOG.info('-' * 75 + 'End Of Test' + '-' * 75)
        self.save(DefaultConfig.get_config('max_iterations'))
        return model_all_values



