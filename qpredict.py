import argparse
from qutils import *
from PIL import Image

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="inference", type=str, metavar='DIR', help='path to get images')

    args = parser.parse_args()
    file_names = sorted(os.listdir(args.path))

    mymodel = transferedModel(120)
    mymodel.to(device)

    os.makedirs(os.path.join("result"), exist_ok=True)
    mymodel.load_state_dict(torch.load(os.path.join("model_weight", 'best_weight.pt'), map_location=device)['model_state_dict'])

    mymodel.eval()
    imgs_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    path = os.path.join(args.path, file_names[0])
    imgs = imgs_transforms(Image.open(path))[None]
    for i in range(1, len(file_names)):
        path = os.path.join(args.path, file_names[i])
        imgs = torch.cat((imgs, imgs_transforms(Image.open(path))[None]), dim=0)
    with torch.no_grad():
        labels = mymodel(imgs)
        _, labels = torch.max(labels, -1)

    file1 = open("result/predictions.txt", "a")
    for i in range(0, len(file_names)):
        s = "({})    img_name:{}     Label:{}   ".format(i, file_names[i], labels[i].item())
        print(s)
        file1.write(s+"\n")
    file1.close()
