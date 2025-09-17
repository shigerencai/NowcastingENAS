
class Modelallvalues(object):
    objectives_best_list = {}
    def __init__(self, identifier):
        self.identifier = identifier
        self.number_of_parameters = None
        self.number_of_all_blocks = None
        self.number_of_conv_blocks = None
        self.number_of_activation_blocks = None
        self.number_of_combination_blocks = None
        self.number_of_add_blocks = None
        self.number_of_sub_blocks = None
        self.number_of_mul_blocks = None

        self.mae_loss = None
        self.mse_loss = None
        self.training_time = None
        self.csi = None
        self.hss = None
        self.ssim = None
        self.accuracy = None
        self.previous_fitness = []


    def is_ptb_fitness_default(self):
        return self.mae_loss and self.training_time and self.mse_loss


    def is_objectives_new_best(self, objectives_list):
        new_best_list = {}
        for objective in objectives_list:
            if objective not in Modelallvalues.objectives_best_list.keys():
                if objective == 'mae_loss':
                    Modelallvalues.objectives_best_list[objective] = self.mae_loss
                    new_best_list[objective] = self.mae_loss
                elif objective == 'mse_loss':
                    Modelallvalues.objectives_best_list[objective] = self.mse_loss
                    new_best_list[objective] = self.mse_loss
                elif objective == 'csi':
                    Modelallvalues.objectives_best_list[objective] = self.csi
                    new_best_list[objective] = self.csi
                elif objective == 'hss':
                    Modelallvalues.objectives_best_list[objective] = self.hss
                    new_best_list[objective] = self.hss
            else:
                if objective == 'mae_loss':
                    if self.mae_loss < Modelallvalues.objectives_best_list[objective]:
                        Modelallvalues.objectives_best_list[objective] = self.mae_loss
                        new_best_list[objective] = self.mae_loss
                elif objective == 'mse_loss':
                    if self.mse_loss < Modelallvalues.objectives_best_list[objective]:
                        Modelallvalues.objectives_best_list[objective] = self.mse_loss
                        new_best_list[objective] = self.mse_loss
                elif objective == 'csi':
                    if self.csi > Modelallvalues.objectives_best_list[objective]:
                        Modelallvalues.objectives_best_list[objective] = self.csi
                        new_best_list[objective] = self.csi
                elif objective == 'hss':
                    if self.hss > Modelallvalues.objectives_best_list[objective]:
                        Modelallvalues.objectives_best_list[objective] = self.hss
                        new_best_list[objective] = self.hss

        return new_best_list

    def show_all_values(self):
        print('-' * 100, '|')
        print(f'model_name = {self.identifier}\t | number_of_parameters = {self.number_of_parameters}\t| training_time = {self.training_time}')
        print('=' * 100, '|')
        print(f'num_all_blocks = {str(self.number_of_all_blocks)}\t | num_conv_blocks = {str(self.number_of_conv_blocks)}\t | num_activation_blocks = {str(self.number_of_activation_blocks)}')
        print('=' * 100, '|')
        print(f'num_combination_blocks = {str(self.number_of_combination_blocks)}\t || num_add_blocks = {str(self.number_of_add_blocks)}\t | num_mul_blocks = {str(self.number_of_mul_blocks)}\t | num_sub_blocks = {str(self.number_of_sub_blocks)}')
        print('=' * 100, '|')
        print(f'train_MSE_loss = {str(self.mae_loss)}\t | valid_MAE_loss = {str(self.mae_loss)}\t | valid_SSIM = {str(self.ssim)}')
        print('=' * 100, '|')
        print(f'csi = {str(self.csi)}\t | hss = {str(self.hss)}\t | accuracy = {str(self.accuracy)}')
        print('-' * 100, '|')

    def get_values_dict(self):
        values_dict = {}
        values_dict['number_of_parameters'] = self.number_of_parameters
        values_dict['number_of_all_blocks'] = self.number_of_all_blocks
        values_dict['number_of_conv_blocks'] = self.number_of_conv_blocks
        values_dict['number_of_activation_blocks'] = self.number_of_activation_blocks
        values_dict['number_of_combination_blocks'] = self.number_of_combination_blocks
        values_dict['number_of_add_blocks'] = self.number_of_add_blocks
        values_dict['number_of_sub_blocks'] = self.number_of_sub_blocks
        values_dict['number_of_mul_blocks'] = self.number_of_mul_blocks
        values_dict['mae_loss'] = self.mae_loss
        values_dict['mse_loss'] = self.mse_loss
        values_dict['training_time'] = self.training_time
        values_dict['csi'] = self.csi
        values_dict['hss'] = self.hss
        values_dict['ssim'] = self.ssim
        values_dict['accuracy'] = self.accuracy
        return values_dict
