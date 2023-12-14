import os
import json
import torch

if __name__ == '__main__':
    with open('./conf/model_test_config.json', 'r') as config_file:
        config = json.load(config_file)
    data_paths = config["data_paths"]
    file_name = config["file_name"]
    model_dict_name = config["model_dict_name"]
    data_path = os.path.join(data_paths,file_name)
    model_path = os.path.join('./conf/model_dicts', model_dict_name)

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

    model = FusionModel().to(device)
    model.load_state_dict(torch.load(model_file_save_path, map_location=torch.device('cpu')))
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for (data_1160, data_440) in zip(test_loader_1160, test_loader_440):
            inputs_1160, labels_1160 = data_1160[0].to(device), data_1160[1].to(device)
            inputs_440, labels_440 = data_440[0].to(device), data_440[1].to(device)
            outputs = model(inputs_1160, inputs_440)
            _, preds = torch.max(outputs, 1)
            total += labels_1160.size(0)
            correct += (preds == labels_1160).sum().item()
            # 将预测结果和真实标签保存到列表中
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels_1160.cpu().numpy())
