import torch 
import torch.nn as nn
import torchvision
from torchvision import models,datasets,transforms
from torchvision.datasets import ImageFolder
from torch import optim
from torch.utils.data import DataLoader,Dataset,random_split
from PIL import Image
import pandas
import os
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MultilabelAUROC



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PercentCenterCrop:
    def __init__(self,keep):
        self.keep=keep
    
    def __call__(self,img):
        w,h=img.size
        new_w=int(w*self.keep)
        new_h=int(h*self.keep)
        left=(w-new_w)//2
        top=(h-new_h)//2
        
        return img.crop((left,top,left+new_w,top+new_h))
    
class_to_index_map={}

class ChestXRayDataSet(Dataset):
    classes=['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
        'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia', 'Pleural_Thickening',
        'Cardiomegaly', 'Nodule', 'Hernia', 'Mass', 'No Finding', 'Other']
    
    def __init__(self,images_dir,labels_csv,transfrom):
        super().__init__()
        self.transform=transfrom
        self.img_paths=[]
        self.labels=[]
        
        df=pandas.read_csv(labels_csv)
        img_map={}
        # print(df.head)
        for folder in os.listdir(images_dir):
            folder_path=os.path.join(images_dir,folder)
            if folder.startswith("images_") and os.path.isdir(folder_path):
                images_subdir=os.path.join(folder_path,"images")

                if os.path.isdir(images_subdir):  
                     for img_name in os.listdir(images_subdir):
                         img_map[img_name]=os.path.join(images_subdir,img_name)

        for img_name in df['Image Index']:
            self.img_paths.append(img_map[img_name])
        
        all_labels=df['Finding Labels'].str.split('|')
        unique_labels=sorted(set([disease for sublist in all_labels for disease in sublist ]))
        self.classes=unique_labels
        self.class_to_index={c:i for i,c in enumerate(self.classes)}
        # print(self.class_to_index)
        class_to_index_map=self.class_to_index
        for diseases in all_labels:
            label_tensor=torch.zeros(len(self.classes))
            for disease in diseases:
                label_tensor[self.class_to_index[disease]]=1
            self.labels.append(label_tensor)
            
        # print(self.labels)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        img=Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform:
            img=self.transform(img)
        label=self.labels[idx]
        return img,label
    
print(class_to_index_map)  
Mytransform=transforms.Compose(
    [
        PercentCenterCrop(keep=1),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        
    ]
)

Mydataset=ChestXRayDataSet(images_dir="E:/DataSets/CuraLink/archive (1)",labels_csv="E:/DataSets/CuraLink/archive (1)/Data_Entry_2017.csv",transfrom=Mytransform)   

class  Model(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        
        self.model=models.densenet121(pretrained=True)
        
        self.model.classifier=nn.Sequential(
            nn.Linear(in_features=1024,out_features=num_classes)
            )

    def forward(self,x):
        x=self.model(x)
        return x

    def save_params(self):
        torch.save(self.model.state_dict(),"ChexNet.pth")
    
    def load_params(self):
        self.model.load_state_dict(torch.load("ChexNet.pth"))
        
model=Model(num_classes=len(Mydataset.classes))
model.to(device)
optimizer=optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4)

criterion=nn.BCEWithLogitsLoss()



def training_one_epoch(loader,optimizer=optimizer,criterion=criterion):
    model.train()
    running_loss=0
    total=0
    batch_no=0

    for images,labels in loader:
        labels=labels.to(device)
        images=images.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss=running_loss+loss.item()
        batch_no=batch_no+1
        batch_size = labels.size(0)
        total += batch_size

        print(
        f"Batch:{batch_no} -> Images processed:{total} | "
        f"Batch Loss:{loss.item():.4f} | "
        f"Avg Loss so far:{running_loss/batch_no:.4f}"
        )
    return running_loss/len(loader)


auroc=MultilabelAUROC(num_labels=len(Mydataset.classes),average=None).to(device)

def validation(loader,optimizer=optimizer,criterion=criterion):
    model.eval()
    running_loss=0
    batch_no=0
    auroc.reset()
    with torch.no_grad():
        for images,labels in loader:
            images=images.to(device)
            labels=labels.to(device).int()
            outputs=model(images)
            loss=criterion(outputs,labels.float())
            running_loss=running_loss+loss.item()
            batch_no=batch_no+1
            probs=torch.sigmoid(outputs)
            auroc.update(probs,labels)
    final_auroc=auroc.compute()
    mean_auroc=final_auroc.mean().item()
    print("Validation AUROC per class:")
    for cls, score in zip(Mydataset.classes, final_auroc):
        print(f"{cls:<22}: {score:.4f}")
    print("Mean AUROC:", mean_auroc)
    return running_loss / len(loader),mean_auroc


def save_checkpoint(epoch, model, optimizer, best_val_auroc, path="checkpoint.pth"):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_auroc": best_val_auroc
    }, path)
    
    
    
def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_auroc = checkpoint["best_val_auroc"]
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch, best_val_auroc
    else:
        return 0, 0.0
    
    


def main():
    train_size=(int)(len(Mydataset)*0.8)
    test_size=len(Mydataset)-train_size
    
    train_data_set,test_data_set=random_split(Mydataset,[train_size,test_size])
    
    train_data_loader=DataLoader(
        dataset=train_data_set,
        shuffle=True,
        batch_size=12,
        num_workers=2
    )
    test_data_loader=DataLoader(
        dataset=test_data_set,
        shuffle=False,
        batch_size=12,
        num_workers=2
    )
    
    start_epoch, best_val_auroc = load_checkpoint(model, optimizer)
    
    epochs = 10

    for epoch in range(start_epoch,epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-------------------------------------------------->" )


        train_loss = training_one_epoch(loader=train_data_loader)
        print(f"Train Loss: {train_loss:.4f}")


        val_loss, val_auroc = validation(loader=test_data_loader)
        print(f"Val Loss  : {val_loss:.4f}")
        print(f"Val AUROC : {val_auroc:.4f}")

 
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            torch.save(model.state_dict(), "best_chexnet_params.pth")
            print("Best model saved ")
        save_checkpoint(epoch, model, optimizer, best_val_auroc)
        print(f"Checkpoint saved for epoch--> {epoch+1}")



if __name__=="__main__":
    main()