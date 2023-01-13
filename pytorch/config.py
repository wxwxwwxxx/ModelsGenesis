import os
import shutil
import csv
class models_genesis_config:
    model = "Unet3D"
    suffix = "bs96_gradclip_fixed"
    exp_name = model + "-" + suffix
    
    # data
    data = "/dataset/lmdb/hospital_646432.lmdb"
    split_yaml = "/dataset/lmdb/hospital_646432.yaml"
    shape = (64, 64, 32)
    hu_min = -1000.0
    hu_max = 1000.0
    scale = 32
    input_rows = 64
    input_cols = 64 
    input_deps = 32
    nb_class = 1
    
    # model pre-training
    verbose = 1
    weights = None
    batch_size = 96
    optimizer = "sgd"
    workers = 48
    #max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 10000
    patience = 50
    lr = 1

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    root_path = "/ckpt/pretrain"
    # root_path = "/tmp"
    model_path = os.path.join(root_path, "Models", exp_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    logs_path = os.path.join(root_path, "Logs", exp_name)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    tboard_path = os.path.join(root_path, "Tboard", exp_name)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class seg_config:
    def __init__(self, suffix="debug",pretrain_weight = None):
        self.model = "Unet3D"
        self.suffix = suffix
        self.exp_name = self.model + "-" + self.suffix

        # data
        self.data = "/dataset/lmdb/ncs_646432.lmdb"
        self.split_yaml = "/dataset/lmdb/ncs_646432.yaml"

        self.shape = (64, 64, 32)

        self.hu_min = -1000.0
        self.hu_max = 1000.0
        self.scale = 32

        self.input_rows = self.shape[0]
        self.input_cols = self.shape[1]
        self.input_deps = self.shape[2]

        # model pre-training
        self.verbose = 1
        # weights = None
        self.weights = pretrain_weight#"/ckpt/pretrain/Models/Unet3D-bs96_gradclip/Genesis_Chest_CT.pt"
        self.batch_size = 24
        self.optimizer = "adam"
        self.workers = 6
        # max_queue_size = workers * 4
        # save_samples = "png"
        self.nb_epoch = 10000
        self.patience = 50
        self.lr = 1e-3

        # image deformation

        self.root_path = "/ckpt/downstream"
        # root_path = "/tmp"
        self.model_path = os.path.join(self.root_path, "Models", self.exp_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.logs_path = os.path.join(self.root_path, "Logs", self.exp_name)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.tboard_path = os.path.join(self.root_path, "Tboard", self.exp_name)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class cls_config:
    def __init__(self, suffix="debug",pretrain_weight = None):
        self.model = "Unet3D"
        self.suffix = suffix
        self.exp_name = self.model + "-" + self.suffix

        # data
        self.data = "/dataset/lmdb/luna16_fpr_64x64x32.lmdb"
        self.train_fold = [0, 1, 2, 3, 4]
        self.valid_fold = [5, 6]
        self.test_fold = [7, 8, 9]
        # hu_min = -1000
        # hu_max = 1000

        self.shape = (64, 64, 32)

        # self.hu_min = -1000.0
        # self.hu_max = 1000.0
        # self.scale = 32
        self.input_rows = self.shape[0]
        self.input_cols = self.shape[1]
        self.input_deps = self.shape[2]

        # model pre-training
        self.verbose = 1
        # weights = None
        self.weights = pretrain_weight#"/ckpt/pretrain/Models/Unet3D-bs96_gradclip/Genesis_Chest_CT.pt"
        self.batch_size = 12
        self.optimizer = "adam"
        self.workers = 6
        # max_queue_size = workers * 4
        # save_samples = "png"
        self.nb_epoch = 10000
        self.patience = 50
        self.lr = 1e-3

        # image deformation

        self.root_path = "/ckpt/downstream_ncc"
        # root_path = "/tmp"
        self.model_path = os.path.join(self.root_path, "Models", self.exp_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.logs_path = os.path.join(self.root_path, "Logs", self.exp_name)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.tboard_path = os.path.join(self.root_path, "Tboard", self.exp_name)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class lcs_config:
    def __init__(self, suffix="debug",pretrain_weight = None):
        self.model = "Unet3D"
        self.suffix = suffix
        self.exp_name = self.model + "-" + self.suffix

        # data
        self.data = "/dataset/lmdb/lits_full_scaled_128x128x16.lmdb"
        # self.valid_index = [23, 45, 16, 78, 111]
        # self.test_index = [17, 40, 98, 67, 122]
        self.train_idx = [n for n in range(0, 100)]
        self.train_idx.append(130)
        self.valid_idx = [n for n in range(100, 115)]
        self.test_idx = [n for n in range(115, 130)]
        self.shape = (128,128,16)

        self.hu_min = -1000.0
        self.hu_max = 1000.0
        self.scale = 32

        self.input_rows = self.shape[0]
        self.input_cols = self.shape[1]
        self.input_deps = self.shape[2]

        # model pre-training
        self.verbose = 1
        # weights = None
        self.weights = pretrain_weight#"/ckpt/pretrain/Models/Unet3D-bs96_gradclip/Genesis_Chest_CT.pt"
        self.batch_size = 24
        self.optimizer = "adam"
        self.workers = 6
        # max_queue_size = workers * 4
        # save_samples = "png"
        self.nb_epoch = 10000
        self.patience = 60
        self.lr = 1e-3

        # image deformation

        self.root_path = "/ckpt/downstream_lcs"
        # root_path = "/tmp"
        self.model_path = os.path.join(self.root_path, "Models", self.exp_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.logs_path = os.path.join(self.root_path, "Logs", self.exp_name)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.tboard_path = os.path.join(self.root_path, "Tboard", self.exp_name)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class bms_config:
    def csv_read(self,csv_list):
        fn_list = []
        for i in csv_list:
            with open(i, "r") as f:
                flist = csv.reader(f)
                for j in flist:
                    fn_list.append(j[0])
        return fn_list
    def __init__(self, suffix="debug",pretrain_weight = None):
        self.model = "Unet3D"
        self.suffix = suffix
        self.exp_name = self.model + "-" + self.suffix

        # data
        self.data = "/dataset/lmdb/brats_240x240x155.lmdb"
        self.train_csv = ["/workspace/keras/downstream_tasks/data/bms/fold_1.csv","/workspace/keras/downstream_tasks/data/bms/fold_2.csv"]
        self.val_csv = ["/workspace/keras/downstream_tasks/data/bms/fold_3.csv"]

        self.train_key_list = self.csv_read(self.train_csv)
        self.val_key_list = self.csv_read(self.val_csv)

        self.shape = (240,240,155)

        self.hu_min = -1000.0
        self.hu_max = 1000.0
        self.scale = 32
        self.valid_count = 4
        self.input_rows = self.shape[0]
        self.input_cols = self.shape[1]
        self.input_deps = self.shape[2]

        # model pre-training
        self.verbose = 1
        # weights = None
        self.weights = pretrain_weight#"/ckpt/pretrain/Models/Unet3D-bs96_gradclip/Genesis_Chest_CT.pt"
        self.batch_size = 16
        self.optimizer = "adam"
        self.workers = 8
        # max_queue_size = workers * 4
        # save_samples = "png"
        self.nb_epoch = 10000
        self.patience = 50
        self.lr = 1e-3

        # image deformation

        self.root_path = "/ckpt/downstream_bms"
        # root_path = "/tmp"
        self.model_path = os.path.join(self.root_path, "Models", self.exp_name)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.logs_path = os.path.join(self.root_path, "Logs", self.exp_name)
        if not os.path.exists(self.logs_path):
            os.makedirs(self.logs_path)

        self.tboard_path = os.path.join(self.root_path, "Tboard", self.exp_name)

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")