import torch
import os
d = torch.load("/ckpt/Genesis_Chest_CT_Backup.pt")
d['state_dict']['down_tr64.ops.0.conv1.weight']=d['state_dict']['down_tr64.ops.0.conv1.weight'].repeat(1,4,1,1,1)
print(d['state_dict']['down_tr64.ops.0.conv1.weight'].size())
torch.save({
    'state_dict': d['state_dict'],
}, os.path.join("/ckpt", "Genesis_Chest_llCT_BMS.pt"))