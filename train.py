import os
import json

if __name__ == '__main__':
    with open('../conf/2layer_config.json', 'r') as config_file:
        config = json.load(config_file)
    train_test = config["train_test"]
    data_paths = config["data_paths"]
    path_1 = os.path.join(os.path.join(data_paths) ,'train')
    path_2 = os.path.join(os.path.join(data_paths ,train_test) ,'test')
    save_path = config["save_path"]
    save_path = os.path.join(save_path, train_test)
    model_save_path = os.path.join(save_path, 'model_dicts')
    csv_save_paths = os.path.join(save_path, 'epoch_data')
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    if not os.path.exists(csv_save_paths):
        os.makedirs(csv_save_paths)
    file_name = config["file_name"]
    # 还差裁剪对应频率的图片，需要再添加一个循环，保存的时候，也要把频率包含进去
    frequency_dic = {999 :[0 ,209] ,1160 :[7 ,32], 440 :[128 ,20], 200 :[170 ,13]}
    path_train = os.path.join(path_1, file_name)
    path_test = os.path.join(path_2, file_name)
    # 数据预处理：您可以根据需要进行调整
    transform_1160  = transforms.Compose([
        CustomCrop(top=frequency_dic[1160][0], left=0, height=frequency_dic[1160][1], width=129),  # 裁剪图片 (这部分被注释掉了，如果您的数据集需要裁剪请打开)
        transforms.Resize((224, 224)),  # 例如，将所有图片都调整到32x32大小
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_440   = transforms.Compose([
        CustomCrop(top=frequency_dic[440][0], left=0, height=frequency_dic[440][1], width=129),  # 裁剪图片 (这部分被注释掉了，如果您的数据集需要裁剪请打开)
        transforms.Resize((224, 224)),  # 例如，将所有图片都调整到32x32大小
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])