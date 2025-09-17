import random
import numpy as np
import imageio.v2 as imageio


def CIKM_sample(batch_size,mode='random', data_type='train', index=None):
    data_path = 'data_provider/CIKM/' + data_type + '/'
    if data_type == 'train':
        if mode == 'random':
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1, 8000)
                img_fold = data_path + 'sample_'+str(sample_index)+'/'
                batch_imgs = []
                for t in range(1,16):
                    img_path = img_fold + 'img_'+str(t)+'.png'
                    img = imageio.imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
    elif data_type == 'validation':
        if index == None:
            raise 'The index needs to be initialized.'
        if index>2001 or index < 1:
            raise 'Index exceeds the range'
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index>2001:
                index = 2001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = data_path + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imageio.imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index==2001:
            return imgs,(index, 0)
        return imgs, (index, b_cup)
    elif data_type == 'test':
        if index == None:
            raise 'The index needs to be initialized.'
        if index > 4001 or index < 1:
            raise 'Index exceeds the range'
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index > 4001:
                index = 4001
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = data_path + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 16):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imageio.imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index == 4001:
            return imgs,(index,0)
        return imgs, (index, b_cup)

    else:
        raise ("Incorrect model approach")


def Shanghai_sample(batch_size, mode='random', data_type='train', index=None):
    data_path = 'data_provider/Shanghai/' + data_type + '/'
    if data_type == 'train':
        if mode == 'random':
            imgs = []
            for batch_idx in range(batch_size):
                sample_index = random.randint(1, 1108)
                img_fold = data_path + 'sample_'+str(sample_index)+'/'
                batch_imgs = []
                for t in range(1, 21):
                    img_path = img_fold + 'img_'+str(t)+'.png'
                    img = imageio.imread(img_path)[:, :, np.newaxis]
                    batch_imgs.append(img)
                imgs.append(np.array(batch_imgs))
            imgs = np.array(imgs)
            return imgs
    elif data_type == 'validation':
        if index == None:
            raise 'The index needs to be initialized'
        if index > 373 or index < 1:
            raise 'Index exceeds the range'
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index > 373:
                index = 373
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = data_path + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 21):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imageio.imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index == 373:
            return imgs, (index, 0)
        return imgs, (index, b_cup)
    elif data_type == 'test':
        if index == None:
            raise 'The index needs to be initialized'
        if index > 741 or index < 1:
            raise 'Index exceeds the range'
        imgs = []
        b_cup = batch_size-1
        for batch_idx in range(batch_size):
            if index > 741:
                index = 741
                b_cup = batch_idx
                imgs.extend([imgs[-1] for _ in range(batch_size-batch_idx)])
                break
            img_fold = data_path + 'sample_'+str(index)+'/'
            batch_imgs = []
            for t in range(1, 21):
                img_path = img_fold + 'img_' + str(t) + '.png'
                img = imageio.imread(img_path)[:, :, np.newaxis]
                batch_imgs.append(img)
            imgs.append(np.array(batch_imgs))
            index = index+1
        imgs = np.array(imgs)
        if index == 741:
            return imgs, (index, 0)
        return imgs, (index, b_cup)

    else:
        raise ("Incorrect model approach")