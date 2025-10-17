import os
import json
import matplotlib.pyplot as plt
from datetime import datetime

def load_loss_from_trainer_state(path):
    with open(path, 'r') as f:
        data = json.load(f)
        loss = [x.get("loss", 0) for x in data['log_history']]
        loss.pop()
    return loss

def llavaplot(directory, destination, **kwargs):
    if kwargs.get('step_pretrain',None) == None: kwargs['step_pretrain'] = 15
    if kwargs.get('start_pretrain',None) == None: kwargs['start_pretrain'] = 100
    if kwargs.get('step_lora', None) == None: kwargs['step_lora'] = 25
    if kwargs.get('start_lora', None) == None: kwargs['start_lora'] = 100

    def get_name(path):
        from pathlib import Path
        path = Path(path)
        return path.parent.parent.name

    fig, axs = plt.subplots(2, 1, figsize=(50, 50))
    axs[0].set_ylabel('Pretrain', rotation=0,labelpad=30,fontsize=30)
    axs[1].set_ylabel('Lora', rotation=0, labelpad=20,fontsize=30)
    def plotllava(subplot, path, **kwargs):
        loss = load_loss_from_trainer_state(path)[kwargs['start_pretrain']:] if not subplot else load_loss_from_trainer_state(path)[kwargs['start_lora']:]
        idx = list(range(len(loss)))
        step = kwargs['step_pretrain'] if not subplot else kwargs['step_lora']
        axs[subplot].plot(idx[::step],loss[::step], lw=0.5,label=get_name(path))
        axs[subplot].legend(fontsize=30) # 很关键，不然不会显示图例
        

    for item in os.listdir(directory):
        _ = os.path.join(directory,item,'pretrain','trainer_state.json')
        if os.path.exists(_): plotllava(0,_,**kwargs)
        _ = os.path.join(directory,item,'lora','trainer_state.json')
        if os.path.exists(_): plotllava(1,_,**kwargs)

    plt.savefig(os.path.join(destination, datetime.now().strftime("%Y%m%d_%H%M")+".png"))
    # plt.show()

if __name__ == '__main__':
    llavaplot(R'C:\Users\admin\Desktop',R'C:\Users\admin\Desktop',
              step_pretrain=15,
              start_pretrain=100,
              step_lora=25,
              start_lora=100)
    