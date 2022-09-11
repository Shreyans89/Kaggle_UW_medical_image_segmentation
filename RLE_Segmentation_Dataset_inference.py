
import sklearn
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset,DataLoader
import pdb




## dataset class gto nload only X (images), to generate dataloaders for inference
class RLE_SegmentationDataset_inference(Dataset):
    def __init__(self,img_paths_df,  transform=transforms.Resize((224,224))
                 ):
        
        self.items=img_paths_df
        self.item_transform = transform
        
        
    def __len__(self):
        return len(self.items)
    
   
    def get_img_T(self,img_path):
        img_t=TensorImage(Image.open(img_path))
        H,W=img_t.shape
        if self.item_transform:
            img_t = self.item_transform(img_t.unsqueeze(0))
            
        return img_t.to(torch.float32) ,H,W                        
    
    def __getitem__(self, idx):
        
        item_data=self.items.iloc[idx]
        img_path,img_id=item_data['img_path'],item_data['id']
        img_t,H,W=self.get_img_T(img_path)
        
       ## return 3 channel image to put it through the fastai_model
        return img_t*torch.ones((3,1,1))
    
    
## dataset class to load both X and Y (for training)  - modify only __getitem__ method
class RLE_SegmentationDataset_train(Dataset):
    
    def __init__(self,img_paths_df, label_df=pd.read_csv('train.csv'), item_transform=transforms.Resize((224,224)), 
                 target_transform=transforms.Resize((224,224))):
        
        self.items=img_paths_df
        self.inference_ds=RLE_SegmentationDataset_inference(img_paths_df,item_transform)
        self.img_labels_df=label_df
        self.target_transform = target_transform
        self.codes={'background':0,'small_bowel':3,'large_bowel':2,'stomach':1}
        
        
        
    def __len__(self):
        return len(self.items)
    
    
    def get_mask_T(self,img_id,H,W):
        
        label_data=self.img_labels_df[self.img_labels_df['id']==img_id]
        label_data=label_data[label_data['segmentation'].notnull()]
         #Incase of no segmentation rle encoding, assert that every pixel belongs to the background
        if len(label_data)==0:
            mask_t=torch.zeros((H,W))
        else: 
            ## in case of overlap b/w pixels of different classes in labels,assign to
            ## smallest class in order small_bowel<large_bowel<stomach
            mask_t=torch.Tensor(np.stack([self.rle2mask(rle_pixels,H,W)*self.codes[pxl_cls] 
                                    for pxl_cls,rle_pixels in 
              zip(label_data['class'],label_data['segmentation'])]).max(axis=0))
        if self.target_transform:
            mask_t = TensorMask(self.target_transform(mask_t.unsqueeze(0)).squeeze(0).to(torch.int64))
          
        return mask_t   
   


         
    def foreground_only(self):
        """Get a new dataset with images which have foreground segmentation classes"""
        merge_img_lbls=pd.merge(left=self.items,right=self.img_labels_df,how='left')
        nb_ids=np.unique(merge_img_lbls['id'][ merge_img_lbls['segmentation'].notnull()])
       # nb_df=pd.DataFrame({'id': nb_ids})
        foreground_df=self.items.set_index('id').loc[nb_ids]
        return self.__class__(foreground_df.reset_index())
        

    def train_test_split(self,test_pct=0.2,random_seed=42):
        train_df,test_df=sklearn.model_selection.train_test_split(self.items,test_size=test_pct,
                                                                 random_state=random_seed)
        
        return self.__class__(train_df),self.__class__(test_df)
    
    
    
    @staticmethod
    
    def rle2mask(rle_pixels,H,W):
        
        rle_pixels=np.array(rle_pixels.split(sep=' '),dtype='int32')
        
        ## even entries are pixel start values
        pxl_starts=rle_pixels[np.arange(start=0,stop=len(rle_pixels),step=2)]

        ## Odd ones are number of pixels starting from start point in the mask
        pxl_offsets=rle_pixels[np.arange(start=1,stop=len(rle_pixels),step=2)]

        pxl_stops=pxl_starts+pxl_offsets

        ## transpose so that flatten gives pixel locations as in start-stop-start-stop
        pxl_start_locs=(np.stack([pxl_starts,pxl_stops]).T).flatten()

        ## initialize flattened mask
        flat_mask=np.arange(start=0,stop=H*W,step=1)+1

        ## compute mask with pxl start locastions. an even number at each position 
        ## implies background ,odd numbers are foreground

        even_odd_mask=(np.expand_dims(flat_mask,1)>=np.expand_dims(pxl_start_locs,0)).sum(axis=1)

        ## odd numbers are foreground class
        mask=even_odd_mask%2

        mask=mask.reshape(H,W) 
        
        
        return mask
    
    def save_masks(self,folder_name='masks'):
        """create and save seh masks from RLE in 
          a folder 'masks' at the same level as train
          (containing images)
          each mask is identified by its corresponding 
          image file's id (taken from train.csv) """
        
        save_folder=Path(folder_name)
        save_folder.mkdir(exist_ok=True)
        
        for i,img_data in tqdm(enumerate(self.items)):
            _,mask=self[i]
            mask=PILMask.create(mask)
            mask_path=save_folder/Path(img_data['id']+img_data['img_path'].suffix)
            mask.save(mask_path)
            
    
    
    def __getitem__(self, idx):
        
        item_data=self.items.iloc[idx]
        img_path,img_id=item_data['img_path'],item_data['id']
        img_t,H,W=self.inference_ds.get_img_T(img_path)
        
        mask_t=self.get_mask_T(img_id,H,W)
        return img_t, mask_t
    
        
