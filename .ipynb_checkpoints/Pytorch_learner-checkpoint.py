
import pdb
from tqdm.notebook import tqdm


class PyTorchLearner():
    def __init__(self,model,dls,optim_class,lr,
                 loss_function,metrics=None,device='cuda',**kwargs):
        self.device=device
        self.model=model.to(self.device)
        self.dls=dls
        self.lr=lr
        self.optimizer = optim_class(self.model.parameters(),self.lr,**kwargs)
        self.loss_function = loss_function
        self.metrics = metrics
    
    
    @staticmethod
    def update_running_stats(running_val,current_val,num_evals):
        running_val+=(1/(num_evals+1)*(current_val-running_val)).detach().item()
        return running_val

    
    
    def train_epoch(self,train_loader):
        """run one training epoch on train dl return average train loss 
         and optionally run callbacks"""
        self.model.train()
        running_trainloss=0.
      
        for j,(xb,yb) in enumerate(tqdm(train_loader)):
            xb,yb=xb.to(self.device),yb.to(self.device)
            y_pred=self.model(xb)
            trainloss=self.loss_function(y_pred,yb)
            running_trainloss=PyTorchLearner.update_running_stats(running_trainloss,
                                                                     trainloss,j)
            self.optimizer.zero_grad()
            trainloss.backward()
            self.optimizer.step()
        return running_trainloss 
    
     
    def valid_epoch(self,eval_loader):
        """run one evaluation epoch on the eval loader,return valid loss
           and metrics"""
        self.model.eval()
        running_evalloss=0.
        running_metrics=defaultdict(int)
        for k,(xb,yb) in enumerate(tqdm(eval_loader)):
            xb,yb=xb.to(self.device),yb.to(self.device)
            with torch.no_grad():
                y_pred=self.model(xb)
                if self.metrics:
                    for metric_name,metric in self.metrics.items():
                        running_metrics[metric_name]=PyTorchLearner.update_running_stats(running_metrics[metric_name],
                                                                      metric(y_pred,yb),k)
                evalloss=self.loss_function(y_pred,yb)
                running_evalloss=PyTorchLearner.update_running_stats(running_evalloss,
                                                                      evalloss,k)
            return running_evalloss,running_metrics
    
    
    
    
    
    
    def fit(self,n_epochs):
        train_loader,eval_loader=self.dls
        """ run n_epoch training and validation epochs"""
        log=[]
        for i in tqdm(range(n_epochs)):
            train_loss=self.train_epoch(train_loader)
            eval_loss,metrics_dict=self.valid_epoch(eval_loader)
                
            row={'epoch':i, 'train_loss':train_loss, 'eval_loss': eval_loss}
            row.update(metrics_dict)
            wandb.log(row)
            log.append(row)
            tracking_df=pd.DataFrame(log)
            display(tracking_df)
        return tracking_df
    
    
   
