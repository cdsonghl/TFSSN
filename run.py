import os
import json
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn

from core.TFSSN import FusionModel,CustomCrop
from lib.utils import SWJTU_Gear_Data, get_fs, get_split_coordinate

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    with open('./conf/model_test_config.json', 'r') as config_file:
        config = json.load(config_file)
    data_paths = config["data_paths"]
    file_name = config["file_name"]
    label_file_name = config["label_file_name"]
    save_path = config["save_path"]
    model_dict_name = config["model_dict_name"]
    data_path = os.path.join(data_paths,file_name)
    model_path = os.path.join('./conf/model_dicts', model_dict_name)

    TEETH_NUMBER_LIST = [29,95,36,90]
    SPEED = 40
    FS_MAX = 1300

    data= SWJTU_Gear_Data(filename=file_name, label_file_name = label_file_name, file_path=data_paths)
    for data_id in ['1','2','3','4']:
        l_t, l_f = data.plt_stft(data_id, save_path, FS_MAX)

    fs_list = get_fs(TEETH_NUMBER_LIST, SPEED)
    fs_dict = get_split_coordinate(fs_list, FS_MAX, l_f)
    # 数据预处理：您可以根据需要进行调整
    transform_fs_0  = transforms.Compose([
        CustomCrop(top=fs_dict['fs_0'][0], left=0, height=fs_dict['fs_0'][1], width=l_t),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_fs_1   = transforms.Compose([
        CustomCrop(top=fs_dict['fs_1'][0], left=0, height=fs_dict['fs_1'][1], width=l_t),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset_fs_0 = torchvision.datasets.ImageFolder(root=save_path, transform=transform_fs_0)
    test_dataset_fs_1 = torchvision.datasets.ImageFolder(root=save_path, transform=transform_fs_1)

    test_loader_fs_0 = torch.utils.data.DataLoader(test_dataset_fs_0, batch_size=32, shuffle=False, num_workers=4,pin_memory=True)
    test_loader_fs_1 = torch.utils.data.DataLoader(test_dataset_fs_1, batch_size=32, shuffle=False, num_workers=4,pin_memory=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    model = FusionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for (data_fs_0, data_fs_1) in zip(test_loader_fs_0, test_loader_fs_1):
            inputs_fs_0, labels = data_fs_0[0].to(device), data_fs_0[1].to(device)
            inputs_fs_1, _ = data_fs_1[0].to(device), data_fs_1[1].to(device)
            outputs = model(inputs_fs_0, inputs_fs_1)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    print("predictions:", predictions)
    print("true_labels:", true_labels)
