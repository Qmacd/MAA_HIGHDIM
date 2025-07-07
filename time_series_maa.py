import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
import seaborn as sns
import numpy as np
from functools import wraps
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from typing import List, Optional
import time
import glob
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import logging
import matplotlib.pyplot as plt
import pandas as pd
import copy
import random
import numpy as np
from abc import ABC, abstractmethod
import models
from tqdm import tqdm  # 添加tqdm进度条

scaler = GradScaler()

class MAABase(ABC):
    """
    MAA 框架的虚基类，定义核心方法接口。
    所有子类必须实现以下方法。
    """

    def __init__(self, N_pairs, batch_size, num_epochs,
                 generator_names, discriminators_names,
                 output_dir,
                 initial_learning_rate = 2e-4,
                 train_split = 0.8,
                 precise = torch.float32,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 device = None,
                 seed=None,
                 ckpt_path="auto",):
        """
        初始化必备的超参数。

        :param N_pairs: 生成器or对抗器的个数
        :param batch_size: 小批次处理
        :param num_epochs: 预定训练轮数
        :param initial_learning_rate: 初始学习率
        :param generators: 建议是一个iterable object，包括了表示具有不同特征的生成器
        :param discriminators: 建议是一个iterable object，可以是相同的判别器
        :param ckpt_path: 各模型检查点
        :param output_path: 可视化、损失函数的log等输出路径
        """

        self.N = N_pairs
        self.initial_learning_rate = initial_learning_rate
        self.generator_names = generator_names
        self.discriminators_names = discriminators_names
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_split = train_split
        self.seed = seed
        self.do_distill_epochs = do_distill_epochs
        self.cross_finetune_epochs = cross_finetune_epochs
        self.device = device
        self.precise = precise

        self.set_seed(self.seed)  # 初始化随机种子
        self.device = setup_device(device)
        print("Running Device:", self.device)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("Output directory created! ")


    def set_seed(self, seed):
        """
        设置随机种子以确保实验的可重复性。

        :param seed: 随机种子
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def process_data(self):
        """数据预处理，包括读取、清洗、划分等"""
        pass

    @abstractmethod
    def init_model(self):
        """模型结构初始化"""
        pass

    @abstractmethod
    def init_dataloader(self):
        """初始化用于训练与评估的数据加载器"""
        pass

    @abstractmethod
    def init_hyperparameters(self):
        """初始化训练所需的超参数"""
        pass

    @abstractmethod
    def train(self):
        """执行训练过程"""
        pass

    @abstractmethod
    def save_models(self):
        """执行训练过程"""
        pass

    @abstractmethod
    def distill(self):
        """执行知识蒸馏过程"""
        pass

    @abstractmethod
    def visualize_and_evaluate(self):
        """评估模型性能并可视化结果"""
        pass

    @abstractmethod
    def init_history(self):
        """初始化训练过程中的指标记录结构"""
        pass

def log_execution_time(func):
    """装饰器：记录函数的运行时间，并动态获取函数名"""

    @wraps(func)  # 保留原函数的元信息（如 __name__）
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行目标函数
        end_time = time.time()  # 记录结束时间
        elapsed_time = end_time - start_time  # 计算耗时

        # 动态获取函数名（支持类方法和普通函数）
        func_name = func.__name__
        print(f"MAA_time_series - '{func_name}' elapse time: {elapsed_time:.4f} sec")
        return result

    return wrapper


def generate_labels(y, num_classes=3, method='percentile'):
    """
    生成分类标签，支持两种方法：
    1. percentile: 基于分位数分割（适用于回归任务）
    2. sequential: 基于时间序列比较（适用于分类任务）
    
    参数：
        y: 数组，形状为 (样本数, ) 或 (样本数, 1)
        num_classes: 类别数，支持2或3
        method: 'percentile' 或 'sequential'
    返回：
        labels: 生成的标签数组，长度与 y 相同
    """
    y = np.array(y).flatten()  # 转成一维数组
    
    if method == 'percentile':
        # 基于分位数的标签生成（适用于回归任务）
        if num_classes == 2:
            # 二分类：使用中位数分割
            median = np.percentile(y, 50)
            labels = np.zeros(len(y), dtype=np.int32)
            labels[y <= median] = 0  # 低值
            labels[y > median] = 1   # 高值
            max_label = 1
        elif num_classes == 3:
            # 三分类：使用三分位数分割
            q33 = np.percentile(y, 33.33)
            q67 = np.percentile(y, 66.67)
            labels = np.zeros(len(y), dtype=np.int32)
            labels[y <= q33] = 0  # 低值
            labels[(y > q33) & (y <= q67)] = 1  # 中等值
            labels[y > q67] = 2  # 高值
            max_label = 2
        else:
            raise ValueError(f"Unsupported num_classes: {num_classes} for percentile method")
        
        labels_array = labels
        
    elif method == 'sequential':
        # 基于时间序列比较的标签生成（适用于分类任务）
        if num_classes == 2:
            # 二分类：0=下降/平稳，1=上升
            labels = [0]  # 对于第一个样本，默认为0
            
            for i in range(1, len(y)):
                if y[i] > y[i - 1]:
                    labels.append(1)  # 上涨
                else:
                    labels.append(0)  # 下跌或平稳
                    
            max_label = 1
            
        elif num_classes == 3:
            # 三分类：0=下降，1=平稳，2=上升
            labels = [1]  # 对于第一个样本，默认平稳
            
            for i in range(1, len(y)):
                if y[i] > y[i - 1]:
                    labels.append(2)  # 上涨
                elif y[i] < y[i - 1]:
                    labels.append(0)  # 下跌
                else:
                    labels.append(1)  # 平稳
                    
            max_label = 2
        else:
            raise ValueError(f"Unsupported num_classes: {num_classes} for sequential method")
        
        labels_array = np.array(labels)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'percentile' or 'sequential'")
    
    # 验证标签范围
    if np.any(labels_array < 0) or np.any(labels_array > max_label):
        print(f"[ERROR] Invalid labels generated: min={labels_array.min()}, max={labels_array.max()}")
        # 修正无效标签到有效范围
        labels_array = np.clip(labels_array, 0, max_label)
        print(f"[FIX] Labels clipped to range [0, {max_label}]")
    
    print(f"[DEBUG] Generated {num_classes}-class labels using {method} method: min={labels_array.min()}, max={labels_array.max()}, unique={np.unique(labels_array)}")
    
    return labels_array


class MAA_time_series(MAABase):
    def __init__(self, args, N_pairs: int, batch_size: int, num_epochs: int,
                 generator_names: List, discriminators_names: Optional[List],
                 output_dir: str,
                 window_sizes: int,
                 ERM: bool,
                 target_name:str,
                 initial_learning_rate: float = 2e-5,
                 train_split: float = 0.8,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 precise=torch.float32,
                 device=None,
                 seed: int = None,
                 ckpt_path: str = None,
                 gan_weights=None,
                 ):
        """
        初始化必备的超参数。

        :param N_pairs: 生成器or对抗器的个数
        :param batch_size: 小批次处理
        :param num_epochs: 预定训练轮数
        :param initial_learning_rate: 初始学习率
        :param generator_names: list object，包括了表示具有不同特征的生成器的名称
        :param discriminators_names: list object，包括了表示具有不同判别器的名称，如果没有就不写默认一致
        :param output_path: 可视化、损失函数的log等输出目录
        :param ckpt_path: 预测时保存的检查点
        """
        super().__init__(N_pairs, batch_size, num_epochs,
                         generator_names, discriminators_names,
                         output_dir,
                         initial_learning_rate,
                         train_split,
                         precise,
                         do_distill_epochs, cross_finetune_epochs,
                         device,
                         seed,
                         ckpt_path)  # 调用父类初始化

        self.args = args
        self.window_sizes = window_sizes
        self.ERM = ERM
        self.ckpt_dir=os.path.join(output_dir, 'ckpt')
        self.target_name=target_name
        
        # 获取类别数和任务模式
        self.num_classes = getattr(args, 'maa_num_classes', 3)  # 默认3类
        self.task_mode = getattr(args, 'task_mode', 'regression')  # 获取任务模式
        
        # 根据任务类型选择标签生成方法
        if self.task_mode == 'regression':
            label_method = 'percentile'  # 回归任务使用分位数方法
        else:
            label_method = 'sequential'  # 分类/投资任务使用时间序列比较
            
        print(f"[DEBUG] MAA_time_series initialized with num_classes={self.num_classes}, task_mode={self.task_mode}, label_method={label_method}")
        # 初始化空字典
        self.generator_dict = {}
        self.discriminator_dict = {"default": models.Discriminator3}

        # 遍历 model 模块下的所有属性
        for name in dir(models):
            obj = getattr(models, name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                lname = name.lower()
                if "generator" in lname:
                    key = lname.replace("generator_", "")
                    self.generator_dict[key] = obj
                elif "discriminator" in lname:
                    key = lname.replace("discriminator", "")
                    self.discriminator_dict[key] = obj

        self.gan_weights = gan_weights

        self.init_hyperparameters()

    @log_execution_time
    def process_data(self, data_path, start_row, end_row,  target_columns, feature_columns_list):
        """
        Process the input data by loading, splitting, and normalizing it.

        Args:
            data_path (str): Path to the CSV data file
            target_columns (list): Indices of target columns
            feature_columns (list): Indices of feature columns

        Returns:
            tuple: (train_x, test_x, train_y, test_y, y_scaler)
        """
        print(f"Processing data with seed: {self.seed}")  # Using self.seed

        # Load data
        data = pd.read_csv(data_path)

        # Select target columns
        y = data.iloc[start_row:end_row, target_columns].values
        target_column_names = data.columns[target_columns]
        print("Target columns:", target_column_names)


        # # Select feature columns
        # x = data.iloc[start_row:end_row, feature_columns].values
        # feature_column_names = data.columns[feature_columns]
        # print("Feature columns:", feature_column_names)

        # Process each set of feature columns
        x_list = []
        feature_column_names_list = []
        self.x_scalers = []  # Store multiple x scalers

        for feature_columns in feature_columns_list:
            # Select feature columns
            x = data.iloc[start_row:end_row, feature_columns].values
            feature_column_names = data.columns[feature_columns]
            print("Feature columns:", feature_column_names)

            x_list.append(x)
            feature_column_names_list.append(feature_column_names)

        # —— 1. 计算并打印总体 y 的均值和方差 ——
        print(f"Overall  Y mean: {y.mean():.4f}, var: {y.var():.4f}")

        # Data splitting using self.train_split
        train_size = int(data.iloc[start_row:end_row].shape[0] * self.train_split)
        # train_x, test_x = x[:train_size], x[train_size:]
        # Split each x in the list
        train_x_list = [x[:train_size] for x in x_list]
        test_x_list = [x[train_size:] for x in x_list]
        train_y, test_y = y[:train_size], y[train_size:]


        # —— 3. 计算并打印 train 和 test 的均值、方差 ——
        print(f"Train    Y mean: {train_y.mean():.4f}, var: {train_y.var():.4f}")
        print(f"Test     Y mean: {test_y.mean():.4f}, var: {test_y.var():.4f}")

        # Normalize each x set separately
        self.train_x_list = []
        self.test_x_list = []
        for train_x, test_x in zip(train_x_list, test_x_list):
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            self.train_x_list.append(x_scaler.fit_transform(train_x))
            self.test_x_list.append(x_scaler.transform(test_x))
            self.x_scalers.append(x_scaler)  # Store all x scalers

        # Normalization
        self.x_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable
        self.y_scaler = MinMaxScaler(feature_range=(0, 1))  # Store scaler as instance variable

        # self.train_x = self.x_scaler.fit_transform(train_x)
        # self.test_x = self.x_scaler.transform(test_x)

        self.train_y = self.y_scaler.fit_transform(train_y)
        self.test_y = self.y_scaler.transform(test_y)

        # 生成训练集的分类标签（直接在 GPU 上生成）
        self.train_labels = generate_labels(self.train_y, self.num_classes)
        # 生成测试集的分类标签
        self.test_labels = generate_labels(self.test_y, self.num_classes)
        print(self.train_y[:5])
        print(self.train_labels[:5])
        # ------------------------------------------------------------------

    def create_sequences_combine(self, x_list, y, label, window_size, start):
        x_ = []
        y_ = []
        y_gan = []
        label_gan = []
        # Create sequences for each x in x_list
        for x in x_list:
            x_seq = []
            for i in range(start, x.shape[0]):
                tmp_x = x[i - window_size: i, :]
                x_seq.append(tmp_x)
            x_.append(np.array(x_seq))

        # Combine x sequences along feature dimension
        x_ = np.concatenate(x_, axis=-1)

        for i in range(start, y.shape[0]):
            # tmp_x = x[i - window_size: i, :]
            tmp_y = y[i]
            tmp_y_gan = y[i - window_size: i + 1]
            tmp_label_gan = label[i - window_size: i + 1]

            # x_.append(tmp_x)
            y_.append(tmp_y)
            y_gan.append(tmp_y_gan)
            label_gan.append(tmp_label_gan)

        x_ = torch.from_numpy(np.array(x_)).float()
        y_ = torch.from_numpy(np.array(y_)).float()
        y_gan = torch.from_numpy(np.array(y_gan)).float()
        label_gan = torch.from_numpy(np.array(label_gan)).float()
        return x_, y_, y_gan, label_gan

    @log_execution_time
    def init_dataloader(self):
        """初始化用于训练与评估的数据加载器"""

        # Sliding Window Processing
        # 分别生成不同 window_size 的序列数据
        train_data_list = [
            self.create_sequences_combine(self.train_x_list, self.train_y, self.train_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        test_data_list = [
            self.create_sequences_combine(self.test_x_list, self.test_y, self.test_labels, w, self.window_sizes[-1])
            for w in self.window_sizes
        ]

        # 分别提取 x、y、y_gan 并堆叠
        self.train_x_all = [x.to(self.device) for x, _, _, _ in train_data_list]
        self.train_y_all = train_data_list[0][1]  # 所有 y 应该相同，取第一个即可，不用cuda因为要eval
        self.train_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in train_data_list]
        self.train_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in train_data_list]

        self.test_x_all = [x.to(self.device) for x, _, _, _ in test_data_list]
        self.test_y_all = test_data_list[0][1]  # 所有 y 应该相同，取第一个即可，不用cuda因为要eval
        self.test_y_gan_all = [y_gan.to(self.device) for _, _, y_gan, _ in test_data_list]
        self.test_label_gan_all = [label_gan.to(self.device) for _, _, _, label_gan in test_data_list]

        assert all(torch.equal(train_data_list[0][1], y) for _, y, _, _ in train_data_list), "Train y mismatch!"
        assert all(torch.equal(test_data_list[0][1], y) for _, y, _, _ in test_data_list), "Test y mismatch!"

        """
        train_x_all.shape  # (N, N, W, F)  不同 window_size 会导致 W 不一样，只能在 W 相同时用 stack
        train_y_all.shape  # (N,)
        train_y_gan_all.shape  # (3, N, W+1)
        """

        self.dataloaders = []

        for i, (x, y_gan, label_gan) in enumerate(
                zip(self.train_x_all, self.train_y_gan_all, self.train_label_gan_all)):
            shuffle_flag = ("transformer" in self.generator_names[i])  # 最后一个设置为 shuffle=True，其余为 False
            dataloader = DataLoader(
                TensorDataset(x, y_gan, label_gan),
                batch_size=self.batch_size,
                shuffle=shuffle_flag,
                generator=torch.manual_seed(self.seed),
                drop_last=True  # 丢弃最后一个不足 batch size 的数据
            )
            self.dataloaders.append(dataloader)

    def init_model(self,num_cls):
        """模型结构初始化"""
        print(f"[DEBUG] init_model called with num_cls = {num_cls}")
        assert len(self.generator_names) == self.N, "Generators and Discriminators mismatch!"
        assert isinstance(self.generator_names, list)
        for i in range(self.N):
            assert isinstance(self.generator_names[i], str)

        self.generators = []
        self.discriminators = []

        for i, name in enumerate(self.generator_names):
            # 获取对应的 x, y
            x = self.train_x_all[i]
            y = self.train_y_all[i]

            # 初始化生成器
            GenClass = self.generator_dict[name]
            if "transformer" in name:
                gen_model = GenClass(x.shape[-1], output_len=y.shape[-1]).to(self.device)
            else:
                gen_model = GenClass(x.shape[-1], y.shape[-1]).to(self.device)

            self.generators.append(gen_model)

            # 初始化判别器（默认只用 Discriminator3）
            DisClass = self.discriminator_dict[
                "default" if self.discriminators_names is None else self.discriminators_names[i]]
            # 修复：input_dim应该是目标维度y.shape[-1]，而不是window_size
            # Discriminator3期望输入[B, W, target_dim]，其中target_dim是目标变量的维度
            target_dim = y.shape[-1] if len(y.shape) > 1 else 1  # 处理标量目标的情况
            print(f"[DEBUG] Creating discriminator {i}:")
            print(f"  - x.shape: {x.shape} (input features)")
            print(f"  - y.shape: {y.shape} (target)")
            print(f"  - target_dim: {target_dim}")
            print(f"  - input_dim: {target_dim}, out_size: {target_dim}, num_cls: {num_cls}")
            dis_model = DisClass(input_dim=target_dim, out_size=target_dim, num_cls=num_cls).to(self.device)
            print(f"[DEBUG] Discriminator {i} created with embedding num_embeddings={dis_model.label_embedding.num_embeddings}")
            self.discriminators.append(dis_model)

    def init_hyperparameters(self, ):
        """初始化训练所需的超参数"""
        # 初始化：对角线上为1，其余为0，最后一列为1.0
        self.init_GDweight = []
        for i in range(self.N):
            row = [0.0] * self.N
            row[i] = 1.0
            row.append(1.0)  # 最后一列为 scale
            self.init_GDweight.append(row)

        if self.gan_weights is None:
            # 最终：均分组合，最后一列为1.0
            final_row = [round(1.0 / self.N, 3)] * self.N + [1.0]
            self.final_GDweight = [final_row[:] for _ in range(self.N)]
        else:
            pass

        self.g_learning_rate = self.initial_learning_rate
        self.d_learning_rate = self.initial_learning_rate
        self.adam_beta1, self.adam_beta2 = (0.9, 0.999)
        self.schedular_factor = 0.1
        self.schedular_patience = 16
        self.schedular_min_lr = 1e-7

    def train(self, logger):
        best_acc,best_model_state = train_multi_gan(
            self.args, self.generators, self.discriminators, self.dataloaders,
            self.window_sizes,
            self.train_x_all, self.train_y_all, self.test_x_all,
            self.test_y_all, self.test_label_gan_all,
            self.do_distill_epochs,self.cross_finetune_epochs,
            self.num_epochs,
            self.output_dir,
            self.device,
            init_GDweight=self.init_GDweight,
            final_GDweight=self.final_GDweight,
            logger=logger)


        self.save_models(best_model_state)
        return best_acc



    def save_models(self, best_model_state):
        """
        保存所有 generator 和 discriminator 的模型参数，包含时间戳、模型名称或编号。
        修复版本：处理 best_model_state 中的 None 值
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join(self.ckpt_dir, timestamp)
        gen_dir = os.path.join(ckpt_dir, "generators")
        disc_dir = os.path.join(ckpt_dir, "discriminators")
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(disc_dir, exist_ok=True)

        # 检查并处理 best_model_state 中的 None 值
        valid_states_count = 0
        for i in range(self.N):
            if best_model_state[i] is not None:
                # 加载最佳模型状态
                self.generators[i].load_state_dict(best_model_state[i])
                valid_states_count += 1
            else:
                # 如果没有最佳状态，使用当前状态并记录警告
                print(f"⚠️  Warning: No best model state for generator {i+1}, using current state")
                logging.warning(f"No best model state for generator {i+1}, using current state")
            
            # 设为评估模式
            self.generators[i].eval()

        # 保存所有生成器
        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(gen_dir, f"{i + 1}_{gen_name}.pt")
            torch.save(gen.state_dict(), save_path)

        # 保存所有判别器
        for i, disc in enumerate(self.discriminators):
            disc_name = type(disc).__name__
            save_path = os.path.join(disc_dir, f"{i + 1}_{disc_name}.pt")
            torch.save(disc.state_dict(), save_path)

        print(f"All models saved with timestamp and identifier.")
        print(f"Valid best states used: {valid_states_count}/{self.N}")
        if valid_states_count < self.N:
            print(f"⚠️  Warning: {self.N - valid_states_count} models saved with current state instead of best state")
            logging.warning(f"{self.N - valid_states_count} models saved with current state instead of best state")

    def get_latest_ckpt_folder(self):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        print(self.ckpt_dir)
        all_subdirs = [d for d in glob.glob(os.path.join(self.ckpt_dir, timestamp[0] + "*")) if os.path.isdir(d)]
        if not all_subdirs:
            raise FileNotFoundError("❌ No checkpoint records!!")
        latest = max(all_subdirs, key=os.path.getmtime)
        print(f"[OK] Auto loaded checkpoint file: {latest}")
        return latest

    def load_model(self):
        gen_path = os.path.join(self.ckpt_path, "g{gru}", "generator.pt")
        if os.path.exists(gen_path):
            self.generators[0].load_state_dict(torch.load(gen_path, map_location=self.device))
            print(f"✅ Loaded generator from {gen_path}")
        else:
            raise FileNotFoundError(f"❌ Generator checkpoint not found at: {gen_path}")

    def pred(self):
        if self.ckpt_path == "latest":
            self.ckpt_path = self.get_latest_ckpt_folder()

        print("Start predicting with all generators..")
        best_model_state = [None for _ in range(self.N)]
        current_path = os.path.join(self.ckpt_path, "generators")

        for i, gen in enumerate(self.generators):
            gen_name = type(gen).__name__
            save_path = os.path.join(current_path, f"{i + 1}_{gen_name}.pt")
            state_dict = torch.load(save_path, map_location=self.device, weights_only=True)
            gen.load_state_dict(state_dict)
            best_model_state[i] = state_dict

        csv_save_dir = self.output_dir
        test_csv_dir = os.path.join(csv_save_dir, "test")
        if not os.path.exists(test_csv_dir):
            os.makedirs(test_csv_dir)
        train_csv_dir = os.path.join(csv_save_dir, "train")
        if not os.path.exists(train_csv_dir):
            os.makedirs(train_csv_dir)
        # —— 新增：遍历每个 generator，把“归一化后”->“原始价格”的真实/预测值保存到 CSV ——
        with torch.no_grad():
            for i, gen in enumerate(self.generators):
                gen.eval()
                # 准备输入、真实 y
                x_test = self.test_x_all[i]  # Tensor on device, shape=(N, W, F)
                y_true_norm = self.test_y_all.cpu().numpy()  # shape=(N,)
                # 前向预测（归一化后）
                y_pred_norm = gen(x_test)[0].cpu().numpy().reshape(-1, 1)  # (N,1)
                # 反归一化回原始值
                y_true = self.y_scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
                y_pred = self.y_scaler.inverse_transform(y_pred_norm).flatten()

                df = pd.DataFrame({
                    'true': y_true,
                    'pred': y_pred
                })
                test_csv_path = os.path.join(test_csv_dir,f'{self.generator_names[i]}.csv')
                df.to_csv(test_csv_path, index=False)
                print(f"Saved true vs pred for generator {self.generator_names[i]} at: {test_csv_path}")

        test_csv_paths = glob.glob(os.path.join(test_csv_dir, '*.csv'))
        all_true_series, pred_series_list, pred_labels = read_and_collect_data(test_csv_paths)
        if self.N>1:
        # 绘制密度图
            plot_density(all_true_series, pred_series_list, pred_labels, self.output_dir, alpha=0.4,
                         no_grid=True,mode='test',target_name=self.target_name)

        with torch.no_grad():
            for i, gen in enumerate(self.generators):
                gen.eval()
                # 准备输入、真实 y
                x_train = self.train_x_all[i]  # Tensor on device, shape=(N, W, F)
                y_true_norm = self.train_y_all.cpu().numpy()  # shape=(N,)
                # 前向预测（归一化后）
                y_pred_norm = gen(x_train)[0].cpu().numpy().reshape(-1, 1)  # (N,1)
                # 反归一化回原始值
                y_true = self.y_scaler.inverse_transform(y_true_norm.reshape(-1, 1)).flatten()
                y_pred = self.y_scaler.inverse_transform(y_pred_norm).flatten()

                df = pd.DataFrame({
                    'true': y_true,
                    'pred': y_pred
                })

                train_csv_path = os.path.join(train_csv_dir,f'{self.generator_names[i]}.csv')
                df.to_csv(train_csv_path, index=False)
                print(f"Saved true vs pred for generator {self.generator_names[i]} at: {train_csv_path}")

        train_csv_paths = glob.glob(os.path.join(train_csv_dir, '*.csv'))
        all_true_series, pred_series_list, pred_labels = read_and_collect_data(train_csv_paths)

        # 绘制密度图
        if self.N > 1:
            plot_density(all_true_series, pred_series_list, pred_labels, self.output_dir, alpha=0.4,
                         no_grid=True,mode='train',target_name=self.target_name)


        results = evaluate_best_models(self.generators, best_model_state, self.train_x_all, self.train_y_all,
                                       self.test_x_all, self.test_y_all, self.y_scaler,
                                       self.output_dir,self.generator_names,self.target_name,self.ERM)
        return results

    def distill(self):
        """评估模型性能并可视化结果"""
        pass

    def visualize_and_evaluate(self):
        """评估模型性能并可视化结果"""
        pass

    def init_history(self):
        """初始化训练过程中的指标记录结构"""
        pass




def validate(model, val_x, val_y):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁止计算梯度
        val_x = val_x.clone().detach().float()

        # 检查val_y的类型，如果是numpy.ndarray则转换为torch.Tensor
        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y).float()
        else:
            val_y = val_y.clone().detach().float()

        # 使用模型进行预测
        predictions, logits  = model(val_x)
        predictions = predictions.cpu().numpy()
        val_y = val_y.cpu().numpy()

        # 计算均方误差（MSE）作为验证损失
        mse_loss = F.mse_loss(torch.tensor(predictions).float().squeeze(), torch.tensor(val_y).float().squeeze())

        return mse_loss

def validate_with_label(model, val_x, val_y, val_labels):
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁止计算梯度
        val_x = val_x.clone().detach().float()

        # 检查val_y的类型，如果是numpy.ndarray则转换为torch.Tensor
        if isinstance(val_y, np.ndarray):
            val_y = torch.tensor(val_y).float()
        else:
            val_y = val_y.clone().detach().float()

        # labels 用于分类
        if isinstance(val_labels, np.ndarray):
            val_lbl_t = torch.tensor(val_labels).long().to(val_x.device)
        else:
            val_lbl_t = val_labels.clone().detach().long().to(val_x.device)

        # 使用模型进行预测
        predictions, logits  = model(val_x)
        predictions = predictions.cpu().numpy()
        val_y = val_y.cpu().numpy()

        # 计算均方误差（MSE）作为验证损失
        mse_loss = F.mse_loss(torch.tensor(predictions).float().squeeze(), torch.tensor(val_y).float().squeeze())

        true_cls = val_lbl_t[:, -1].squeeze()  # [B]
        pred_cls = logits.argmax(dim=1)  # [B]
        acc = (pred_cls == true_cls).float().mean()  # 标量

        return mse_loss, acc


def plot_generator_losses(data_G, output_dir):
    """
    绘制 G1、G2、G3 的损失曲线。

    Args:
        data_G1 (list): G1 的损失数据列表，包含 [histD1_G1, histD2_G1, histD3_G1, histG1]。
        data_G2 (list): G2 的损失数据列表，包含 [histD1_G2, histD2_G2, histD3_G2, histG2]。
        data_G3 (list): G3 的损失数据列表，包含 [histD1_G3, histD2_G3, histD3_G3, histG3]。
    """

    plt.rcParams.update({'font.size': 12})
    all_data = data_G
    N = len(all_data)
    plt.figure(figsize=(6 * N, 5))

    for i, data in enumerate(all_data):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            plt.plot(acc, label=f"G{i + 1} vs D{j + 1}" if j < N - 1 else f"G{i + 1} Combined", linewidth=2)

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(f"G{i + 1} Loss over Epochs", fontsize=16)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "generator_losses.png"), dpi=500)
    plt.close()


def plot_discriminator_losses(data_D, output_dir):
    plt.rcParams.update({'font.size': 12})
    N = len(data_D)
    plt.figure(figsize=(6 * N, 5))

    for i, data in enumerate(data_D):
        plt.subplot(1, N, i + 1)
        for j, acc in enumerate(data):
            plt.plot(acc, label=f"D{i + 1} vs G{j + 1}" if j < len(data)-1 else f"D{i + 1} Combined", linewidth=2)

        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Loss", fontsize=14)
        plt.title(f"D{i + 1} Loss over Epochs", fontsize=16)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "discriminator_losses.png"), dpi=500)
    plt.close()


def visualize_overall_loss(histG, histD, output_dir):
    plt.rcParams.update({'font.size': 12})
    N = len(histG)
    plt.figure(figsize=(5 * N, 4))

    for i, (g, d) in enumerate(zip(histG, histD)):
        plt.plot(g, label=f"G{i + 1} Loss", linewidth=2)
        plt.plot(d, label=f"D{i + 1} Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.title("Generator & Discriminator Loss", fontsize=16)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_losses.png"), dpi=500)
    plt.close()


def plot_mse_loss(hist_MSE_G, hist_val_loss, num_epochs,
                  output_dir):
    """
    绘制训练过程中和验证集上的MSE损失变化曲线

    参数：
    hist_MSE_G1, hist_MSE_G2, hist_MSE_G3 : 训练过程中各生成器的MSE损失
    hist_val_loss1, hist_val_loss2, hist_val_loss3 : 验证集上各生成器的MSE损失
    num_epochs : 训练的epoch数
    """
    plt.rcParams.update({'font.size': 12})
    N = len(hist_MSE_G)
    plt.figure(figsize=(5 * N, 4))

    for i, (MSE, val_loss) in enumerate(zip(hist_MSE_G, hist_val_loss)):
        plt.plot(range(num_epochs), MSE, label=f"Train MSE G{i + 1}", linewidth=2)
        plt.plot(range(num_epochs), val_loss, label=f"Val MSE G{i + 1}", linewidth=2, linestyle="--")

    plt.title("MSE Loss for Generators (Train vs Validation)", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("MSE", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mse_losses.png"), dpi=500)
    plt.close()

def inverse_transform(predictions, scaler):
    """ 使用y_scaler逆转换预测结果 """
    return scaler.inverse_transform(predictions)


def compute_metrics(true_values, predicted_values):
    """计算MSE, MAE, RMSE, MAPE"""
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100
    per_target_mse = np.mean((true_values - predicted_values) ** 2, axis=0)  # 新增
    return mse, mae, rmse, mape, per_target_mse


def plot_fitting_curve(true_values, predicted_values, output_dir, model_name,target_name):
    """绘制拟合曲线并保存结果"""
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 8))
    plt.plot(true_values, label='True Values', linewidth=2)
    plt.plot(predicted_values, label='Predicted Values', linewidth=2, linestyle='--')
    plt.title(f'{model_name} on {target_name}', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{target_name}_{model_name}_fitting_curve.png', dpi=500)
    plt.close()


def save_metrics(metrics, output_dir, model_name):
    """保存MSE, MAE, RMSE, MAPE到文件"""
    with open(f'{output_dir}/{model_name}_metrics.txt', 'w') as f:
        f.write("MSE: {}\n".format(metrics[0]))
        f.write("MAE: {}\n".format(metrics[1]))
        f.write("RMSE: {}\n".format(metrics[2]))
        f.write("MAPE: {}\n".format(metrics[3]))




def evaluate_best_models(generators, best_model_state, train_xes, train_y, test_xes, test_y, y_scaler, output_dir,
                         generator_names,target_name,ERM):
    N = len(generators)

    for i in range(N):
        generators[i].load_state_dict(best_model_state[i])
        generators[i].eval()

    train_y_inv = inverse_transform(train_y, y_scaler)
    test_y_inv = inverse_transform(test_y, y_scaler)


    train_results = []
    test_results = []

    with torch.no_grad():
        train_fitting_curve_dir=os.path.join(output_dir, "train")
        test_fitting_curve_dir=os.path.join(output_dir, "test")
        for i in range(N):
            if N>1:
                name = f'MAA-TSF-{generator_names[i]}'
            elif not ERM:
                name = f'GAN-{generator_names[i]}'
            else:
                name = f'ERM-{generator_names[i]}'

            # Train
            train_pred, _ = generators[i](train_xes[i])
            train_pred = train_pred.cpu().numpy()
            train_pred_inv = inverse_transform(train_pred, y_scaler)
            train_metrics = compute_metrics(train_y_inv, train_pred_inv)
            plot_fitting_curve(train_y_inv, train_pred_inv,train_fitting_curve_dir, name,target_name)

            # Test
            test_pred, _ = generators[i](test_xes[i])
            test_pred = test_pred.cpu().numpy()
            test_pred_inv = inverse_transform(test_pred, y_scaler)
            test_metrics = compute_metrics(test_y_inv, test_pred_inv)
            plot_fitting_curve(test_y_inv, test_pred_inv, test_fitting_curve_dir, name,target_name)

            # Logging
            print(f"[Train] {name}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, "
                  f"RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")
            print(f"[Test]  {name}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, "
                  f"RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")

            logging.info(f"[Train] {name}: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, "
                         f"RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")
            logging.info(f"[Test]  {name}: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, "
                         f"RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")

            # Collect for external use
            train_results.append({
                "Generator": name,
                "MSE": train_metrics[0],
                "MAE": train_metrics[1],
                "RMSE": train_metrics[2],
                "MAPE": train_metrics[3],
                "MSE_per_target": train_metrics[4].tolist()
            })

            test_results.append({
                "Generator": name,
                "MSE": test_metrics[0],
                "MAE": test_metrics[1],
                "RMSE": test_metrics[2],
                "MAPE": test_metrics[3],
                "MSE_per_target": test_metrics[4].tolist()
            })

    # 不直接保存 CSV，把 train/test results 留给外部处理
    return train_results, test_results


def plot_density(all_true_series, pred_series_list, pred_labels, output_dir, alpha, no_grid,mode,target_name):
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 6))

    # 合并所有真实值数据并绘制其总体的密度分布
    combined_true = pd.concat(all_true_series).dropna()
    if not combined_true.empty:
        sns.kdeplot(combined_true, label='True Value', color='orange',
                    linewidth=1.5, alpha=alpha, fill=True)
    else:
        print("⚠️ No valid real data found, skipping True distribution plot.")

    print("Plotting all prediction distributions...")
    for pred_series, label in zip(pred_series_list, pred_labels):
        if not pred_series.empty:
            sns.kdeplot(pred_series, label=f'Predictions on {label}',
                        linewidth=1.5, alpha=alpha, fill=True)
        else:
            print(f"⚠️ No valid predicted data found in file {label}, skipping plot.")

    ax = plt.gca()
    ax.set(xlabel='Value', ylabel='Density')
    plt.title(f'MAA-TSF on {target_name}', fontsize=16)
    plt.legend()

    if not no_grid:
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()

    out_path = os.path.join(output_dir, f'{target_name}_{mode}_density.png')
    try:
        plt.savefig(out_path)
        print(f"Saved combined plot: {out_path}")
    except Exception as e:
        print(f"❌ Unable to save figure {out_path}: {e}")

    plt.close()

def read_and_collect_data(csv_paths):
    """
    读取所有 CSV 文件并收集数据
    Args:
        csv_paths (list): CSV 文件路径列表

    Returns:
        all_true_series (list): 真实值数据
        pred_series_list (list): 预测值数据
        pred_labels (list): 每个文件的标签
    """
    all_true_series = []
    pred_series_list = []
    pred_labels = []

    print("Reading and collecting data...")

    for path in csv_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"❌ Unable to read file {path}: {e}")
            continue

        if 'true' not in df.columns or 'pred' not in df.columns:
            print(f"⚠️ File {path} missing 'true' or 'pred' column, skipping.")
            continue

        all_true_series.append(df['true'].dropna())
        pred_series_list.append(df['pred'].dropna())
        pred_labels.append(filename)  # 使用文件名作为预测分布的标签

    if not all_true_series:
        print("❌ No valid data found in any file.")
        exit(1)

    return all_true_series, pred_series_list, pred_labels





def setup_device(device):
    if isinstance(device, list) and len(device) == 1:
        device = torch.device(f'cuda:{device[0]}')
    else:
        device = None

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA:", torch.cuda.get_device_name(0))
        elif torch.backends.mps.is_available():
            device = torch.device("mps")  # For Apple Silicon
            print("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("Using CPU")

    return device

def get_autocast_context(amp_dtype: str):
    if amp_dtype == "float16":
        return autocast(dtype=torch.float16)
    elif amp_dtype == "bfloat16":
        return autocast(dtype=torch.bfloat16)
    elif amp_dtype == "mixed":
        return autocast()
    else:
        # 返回一个 dummy context manager，不使用 AMP
        from contextlib import nullcontext
        return nullcontext()


def compute_logdiff(arr):
    """
    对 arr（shape=(T, D) 或 (T,)）做 ln((x_t - x_{t-1}) / x_{t-1})，
    并补齐第一个点为 0，防止 NaN/Inf。
    返回 shape 与输入相同。
    """
    eps = 1e-8
    a = np.array(arr, dtype=float)
    # 差分并防止除零
    diff = (a[1:] - a[:-1]) / (a[:-1] + eps)
    logdiff = np.log(diff + eps)
    # 补齐第一个时间点
    if a.ndim == 1:
        pad = np.array([0.0])
    else:
        pad = np.zeros((1, a.shape[1]), dtype=float)
    out = np.concatenate([pad, logdiff], axis=0)

    # 这里为了简单，我们统计替换后等于 0 的值总数，以及替换前的 NaN：
    nan_count = np.isnan(np.concatenate([pad, logdiff], axis=0)).sum()
    zero_count = (out == 0.0).sum()

    print(f"[compute_logdiff] NaN count (before replacement): {nan_count}, 0 value count (after replacement): {zero_count}")

    # 将任何 NaN 或 Inf 全部替换为 0
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)



def train_multi_gan(args,
                    generators, discriminators,
                    dataloaders,
                    window_sizes,
                    train_xes, train_y,
                    val_xes, val_y, val_labels,
                    distill_epochs, cross_finetune_epochs,
                    num_epochs,
                    output_dir,
                    device,
                    init_GDweight=[
                        [1, 0, 0, 1.0],  # alphas_init
                        [0, 1, 0, 1.0],  # betas_init
                        [0., 0, 1, 1.0]  # gammas_init...
                    ],
                    final_GDweight=[
                        [0.333, 0.333, 0.333, 1.0],  # alphas_final
                        [0.333, 0.333, 0.333, 1.0],  # betas_final
                        [0.333, 0.333, 0.333, 1.0]  # gammas_final...,
                    ],
                    logger=None,
                    dynamic_weight = False):
    N = len(generators)

    assert N == len(discriminators)
    assert N == len(window_sizes)
    assert N >= 1

    g_learning_rate = 1e-4  # 提高学习率
    d_learning_rate = 1e-4  # 提高学习率

    # 二元交叉熵【损失函数，可能会有问题
    # criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    optimizers_G = [torch.optim.AdamW(model.parameters(), lr=g_learning_rate, betas=(0.9, 0.999))
                    for model in generators]

    # 为每个优化器设置 ReduceLROnPlateau 调度器
    schedulers = [lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=16, min_lr=1e-7)
                  for optimizer in optimizers_G]

    optimizers_D = [torch.optim.Adam(model.parameters(), lr=d_learning_rate, betas=(0.9, 0.999))
                    for model in discriminators]

    best_epoch_mse = [-1 for _ in range(N)]
    best_epoch_acc = [-1 for _ in range(N)]

    # 定义生成历史记录的关键字
    """
    以三个为例，keys长得是这样得的：
    ['G1', 'G2', 'G3', 
    'D1', 'D2', 'D3', 
    'MSE_G1', 'MSE_G2', 'MSE_G3', 
    'val_G1', 'val_G2', 'val_G3', 
    'D1_G1', 'D2_G1', 'D3_G1', 'D1_G2', 'D2_G2', 'D3_G2', 'D1_G3', 'D2_G3', 'D3_G3'
    ]
    """

    keys = []
    g_keys = [f'G{i}' for i in range(1, N + 1)]
    d_keys = [f'D{i}' for i in range(1, N + 1)]
    MSE_g_keys = [f'MSE_G{i}' for i in range(1, N + 1)]
    val_loss_keys = [f'val_G{i}' for i in range(1, N + 1)]
    acc_keys = [f'acc_G{i}' for i in range(1, N + 1)]

    keys.extend(g_keys)
    keys.extend(d_keys)
    keys.extend(MSE_g_keys)
    keys.extend(val_loss_keys)
    keys.extend(acc_keys)

    d_g_keys = []
    for g_key in g_keys:
        for d_key in d_keys:
            d_g_keys.append(d_key + "_" + g_key)
    keys.extend(d_g_keys)

    # 创建包含每个值为np.zeros(num_epochs)的字典
    hists_dict = {key: np.zeros(num_epochs) for key in keys}

    best_mse = [float('inf') for _ in range(N)]
    best_acc = [0.0 for _ in range(N)]

    best_model_state = [None for _ in range(N)]

    patience_counter = 0
    patience = 15
    feature_num = train_xes[0].shape[2]
    target_num = train_y.shape[-1]

    print("start training")
    
    # 创建训练epoch进度条
    epoch_pbar = tqdm(range(num_epochs), desc="MAA Training Epochs", unit="epoch")
    
    for epoch in epoch_pbar:
        epo_start = time.time()

        if epoch < 10:
            weight_matrix = torch.tensor(init_GDweight).to(device)
        elif dynamic_weight:
            # —— 动态计算 G-D weight 矩阵 ——
            # 从上一轮的 validation loss 里拿到每个 G 的损失
            # val_loss_keys = ['val_G1', 'val_G2', ..., 'val_GN']
            losses = torch.stack([
                      torch.tensor(hists_dict[val_loss_keys[i]][epoch - 1])
             for i in range(N)
            ]).to(device)  # shape: [N]

            # 性能 Perf_i = -loss_i，beta 控制“硬度”
            perf = torch.exp(-losses)  # shape: [N]
            probs = perf / perf.sum()  # shape: [N], softmax over generators

            # 构造训练 Generator 时用的 N×N 矩阵：每行都是同一分布
            weight_G = probs.unsqueeze(0).repeat(N, 1)  # shape: [N, N]
            weight_G = weight_G + torch.eye(N, device=device)

            # 构造训练 Discriminator 时的 N×(N+1) 矩阵：最后一列保持 1.0（给真数据）
            ones = torch.ones((N, 1), device=device)
            weight_matrix = torch.cat([weight_G, ones], dim=1)  # shape: [N, N+1]
        else:
            weight_matrix = torch.tensor(final_GDweight).to(device)

        keys = []
        keys.extend(g_keys)
        keys.extend(d_keys)
        keys.extend(MSE_g_keys)
        keys.extend(d_g_keys)

        loss_dict = {key: [] for key in keys}

        # use the gap the equalize the length of different generators
        gaps = [window_sizes[-1] - window_sizes[i] for i in range(N - 1)]

        # 创建批次进度条
        batch_pbar = tqdm(enumerate(dataloaders[-1]), total=len(dataloaders[-1]),
                         desc=f"Epoch {epoch+1}/{num_epochs}", leave=False, unit="batch")

        for batch_idx, (x_last, y_last, label_last) in batch_pbar:
            # TODO: maybe try to random select a gap from the whole time windows
            x_last = x_last.to(device)
            y_last = y_last.to(device)
            label_last = label_last.to(device)
            label_last = label_last.unsqueeze(-1)
            # print(x_last.shape, y_last.shape, label_last.shape)

            X = []
            Y = []
            LABELS = []

            for gap in gaps:
                X.append(x_last[:, gap:, :])
                Y.append(y_last[:, gap:, :])
                LABELS.append(label_last[:, gap:, :].long())
            X.append(x_last.to(device))
            Y.append(y_last.to(device))
            LABELS.append(label_last.to(device).long())

            for i in range(N):
                generators[i].eval()
                discriminators[i].train()

            loss_D, lossD_G = discriminate_fake(args, X, Y, LABELS,
                                                generators, discriminators,
                                                window_sizes, target_num,
                                                criterion, weight_matrix,
                                                device, mode="train_D",
                                                mse_weight=1.0, cls_weight=0.3)

            # 3. 存入 loss_dict
            for i in range(N):
                loss_dict[d_keys[i]].append(loss_D[i].item())

            for i in range(1, N + 1):
                for j in range(1, N + 1):
                    key = f'D{i}_G{j}'
                    loss_dict[key].append(lossD_G[i - 1, j - 1].item())

            # 根据批次的奇偶性交叉训练两个GAN
            # if batch_idx% 2 == 0:
            for optimizer_D in optimizers_D:
                optimizer_D.zero_grad()

            # TODO: to see whether there is need to add together


            scaler.scale(loss_D.sum(dim=0)).backward()

            for i in range(N):
                # optimizers_D[i].step()
                scaler.step(optimizers_D[i])
                scaler.update()

                discriminators[i].eval()
                generators[i].train()

            '''训练生成器'''
            weight = weight_matrix[:, :-1].clone().detach()  # [N, N]

            loss_G, loss_mse_G = discriminate_fake(args, X, Y, LABELS,
                                                   generators, discriminators,
                                                   window_sizes, target_num,
                                                   criterion, weight,
                                                   device,
                                                   mode="train_G",
                                                   mse_weight=1.0, cls_weight=0.3)

            for i in range(N):
                loss_dict[g_keys[i]].append(loss_G[i].item())
                loss_dict["MSE_" + g_keys[i]].append(loss_mse_G[i].item())

            for optimizer_G in optimizers_G:
                optimizer_G.zero_grad()


            scaler.scale(loss_G).sum(dim=0).backward()

            for optimizer_G in optimizers_G:
                # optimizer_G.step()
                scaler.step(optimizer_G)
                scaler.update()

        for key in loss_dict.keys():
            hists_dict[key][epoch] = np.mean(loss_dict[key])

        improved = [False] * 3
        for i in range(N):

            hists_dict[val_loss_keys[i]][epoch],hists_dict[acc_keys[i]][epoch]  = validate_with_label(generators[i], val_xes[i], val_y, val_labels[i])

            if hists_dict[val_loss_keys[i]][epoch].item() < best_mse[i]:
                best_mse[i] = hists_dict[val_loss_keys[i]][epoch]
                best_model_state[i] = copy.deepcopy(generators[i].state_dict())
                best_epoch_mse[i] = epoch + 1
                improved[i] = True
            if hists_dict[acc_keys[i]][epoch] > best_acc[i]:
                best_acc[i] = hists_dict[acc_keys[i]][epoch]
                best_epoch_acc[i] = epoch + 1

            schedulers[i].step(hists_dict[val_loss_keys[i]][epoch])

        if distill_epochs > 0 and (epoch + 1) % 30 == 0:
            # if distill and patience_counter > 1:
            losses = [hists_dict[val_loss_keys[i]][epoch] for i in range(N)]
            rank = np.argsort(losses)
            print(f"Do distill {distill_epochs} epoch! Distill from G{rank[0] + 1} to G{rank[-1] + 1}")
            logging.info(f"Do distill {distill_epochs} epoch! Distill from G{rank[0] + 1} to G{rank[-1] + 1}")
            for e in range(distill_epochs):
                do_distill(rank, generators, dataloaders, optimizers_G, window_sizes, device)

        if (epoch + 1) % 10 == 0 and cross_finetune_epochs > 0:
            G_losses = [hists_dict[val_loss_keys[i]][epoch] for i in range(N)]
            D_losses = [np.mean(loss_dict[d_keys[i]]) for i in range(N)]
            G_rank = np.argsort(G_losses)
            D_rank = np.argsort(D_losses)
            print(f"Start cross finetune!  So far G{G_rank[0] + 1} with D{D_rank[0] + 1}")
            logging.info(f"Start cross finetune!  So far G{G_rank[0] + 1} with D{D_rank[0] + 1}")
            # if patience_counter > 1:
            for e in range(cross_finetune_epochs):
                for batch_idx, (x_last, y_last, label_last) in enumerate(dataloaders[-1]):
                    x_last = x_last.to(device)
                    y_last = y_last.to(device)
                    label_last = label_last.to(device)
                    label_last = label_last.unsqueeze(-1)
                    # print(x_last.shape, y_last.shape, label_last.shape)

                    X = []
                    Y = []
                    LABELS = []

                    for gap in gaps:
                        X.append(x_last[:, gap:, :])
                        Y.append(y_last[:, gap:, :])
                        LABELS.append(label_last[:, gap:, :].long())
                    X.append(x_last.to(device))
                    Y.append(y_last.to(device))
                    LABELS.append(label_last.to(device).long())
                    cross_best_Gloss = np.inf

                    generators[G_rank[0]].eval()
                    discriminators[D_rank[0]].train()

                    loss_D, lossD_G = discriminate_fake(args, [X[G_rank[0]]], [Y[D_rank[0]]], [LABELS[D_rank[0]]],
                                                        [generators[G_rank[0]]], [discriminators[D_rank[0]]],
                                                        [window_sizes[D_rank[0]]], target_num,
                                                        criterion, weight_matrix[D_rank[0], G_rank[0]],
                                                        device, mode="train_D",
                                                        mse_weight=1.0, cls_weight=0.3)

                    optimizers_D[D_rank[0]].zero_grad()

                    # loss_D.sum(dim=0).backward()
                    scaler.scale(loss_D.sum(dim=0)).backward()
                    # optimizers_D[D_rank[0]].step()
                    scaler.step(optimizers_D[D_rank[0]])
                    scaler.update()

                    discriminators[D_rank[0]].eval()
                    generators[G_rank[0]].train()

                    '''训练生成器'''
                    weight = weight_matrix[:, :-1].clone().detach()  # [N, N]
                    loss_G, loss_mse_G = discriminate_fake(args, [X[G_rank[0]]], [Y[D_rank[0]]], [LABELS[D_rank[0]]],
                                                           [generators[G_rank[0]]], [discriminators[D_rank[0]]],
                                                           [window_sizes[D_rank[0]]], target_num,
                                                           criterion, weight[D_rank[0], G_rank[0]],
                                                           device,
                                                           mode="train_G",
                                                           mse_weight=1.0, cls_weight=0.3)

                    optimizers_G[G_rank[0]].zero_grad()
                    # loss_G.sum(dim=0).backward()
                    scaler.scale(loss_G.sum(dim=0)).backward()
                    # optimizers_G[G_rank[0]].step()
                    scaler.step(optimizers_G[G_rank[0]])
                    scaler.update()

                validate_G_loss, validate_G_acc = validate_with_label(generators[G_rank[0]], val_xes[G_rank[0]], val_y, val_labels[G_rank[0]])


                if validate_G_loss >= cross_best_Gloss:
                    generators[G_rank[0]].load_state_dict(best_model_state[G_rank[0]])
                    break
                elif validate_G_loss < cross_best_Gloss:
                    cross_best_Gloss = validate_G_loss
                    best_mse[G_rank[0]] = cross_best_Gloss
                    best_model_state[G_rank[0]] = copy.deepcopy(generators[G_rank[0]].state_dict())
                    best_epoch_mse[G_rank[0]] = epoch + 1

                print(
                    f"== Cross finetune Epoch [{e + 1}/{num_epochs}]: G{G_rank[0] + 1} with D{D_rank[0] + 1}: Validation MSE {validate_G_loss:.8f}, Validation Acc {validate_G_acc*100:.2f}%")
                logging.info(
                    f"== Cross finetune Epoch [{e + 1}/{num_epochs}]: G{G_rank[0] + 1} with D{D_rank[0] + 1}: Validation MSE {validate_G_loss:.8f}, Validation Acc {validate_G_acc*100:.2f}%")  # NEW



        # 动态生成打印字符串
        log_str_mse = ", ".join(
            f"G{i + 1}: {hists_dict[key][epoch]:.8f}"
            for i, key in enumerate(val_loss_keys)
        )
        log_str_acc = ", ".join(
            f"G{i + 1}: {hists_dict[key][epoch]*100:.2f} %"
            for i, key in enumerate(acc_keys)
        )
        # if len(acc_keys) == 1:
        #     best_info = ", ".join([f"G{i + 1}:{best_epoch_mse[i]}" for i in range(N)])
        # else:
        #     best_info = ", ".join([f"G{i + 1}:{best_epoch_mse[i]}" for i in range(N)])


        logging.info("Epoch %d | Validation MSE: %s | Accuracy: %s", epoch + 1, log_str_mse, log_str_acc)  # NEW
        #print(f"Patience Counter:{patience_counter}, Best MSE Epochs | {best_info}")
        print(f"Patience Counter:{patience_counter}/{patience}")
        if not any(improved):
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

        epo_end = time.time()
        print(f"Epoch time: {epo_end - epo_start:.4f}")

    data_G = [[[] for _ in range(4)] for _ in range(N)]
    data_D = [[[] for _ in range(4)] for _ in range(N)]

    for i in range(N):
        for j in range(N + 1):
            if j < N:
                data_G[i][j] = hists_dict[f"D{j + 1}_G{i + 1}"][:epoch]
                data_D[i][j] = hists_dict[f"D{i + 1}_G{j + 1}"][:epoch]
            elif j == N:
                data_G[i][j] = hists_dict[g_keys[i]][:epoch]
                data_D[i][j] = hists_dict[d_keys[i]][:epoch]

    plot_generator_losses(data_G, output_dir)
    plot_discriminator_losses(data_D, output_dir)

    # overall G&D
    visualize_overall_loss([data_G[i][N] for i in range(N)], [data_D[i][N] for i in range(N)], output_dir)

    hist_MSE_G = [[] for _ in range(N)]
    hist_val_loss = [[] for _ in range(N)]
    for i in range(N):
        hist_MSE_G[i] = hists_dict[f"MSE_G{i + 1}"][:epoch]
        hist_val_loss[i] = hists_dict[f"val_G{i + 1}"][:epoch]

    plot_mse_loss(hist_MSE_G, hist_val_loss, epoch, output_dir)


    mse_info = ", ".join([f"G{i + 1}:{best_epoch_mse[i]}" for i in range(N)])
    acc_info = ", ".join([f"G{i + 1}:{best_epoch_acc[i]}" for i in range(N)])
    acc_value_info = ", ".join([f"G{i + 1}: {best_acc[i] * 100:.2f}%" for i in range(N)])
    mse_value_info = ", ".join([f"G{i + 1}: {best_mse[i]:.6f}" for i in range(N)])

    print(f"[Best MSE Epochs]     {mse_info}")
    print(f"[Best MSE Values]     {mse_value_info}")
    print(f"[Best ACC Epochs]     {acc_info}")
    print(f"[Best Accuracy Values]{acc_value_info}")

    logging.info(f"[Best MSE Epochs]     {mse_info}")
    logging.info(f"[Best MSE Values]     {mse_value_info}")
    logging.info(f"[Best ACC Epochs]     {acc_info}")
    logging.info(f"[Best Accuracy Values]{acc_value_info}")

    # Ensure best_model_state is never None - use current state if no improvement
    for i in range(N):
        if best_model_state[i] is None:
            print(f"⚠️  Warning: No improvement for generator {i+1}, using current state")
            best_model_state[i] = copy.deepcopy(generators[i].state_dict())

    return best_acc,best_model_state


def discriminate_fake(args, X, Y, LABELS,
                      generators, discriminators,
                      window_sizes, target_num,
                      criterion, weight_matrix,
                      device,
                      mode,
                      mse_weight=1.0,
                      cls_weight=0.3):
    """
    修复版本的discriminate_fake函数
    主要修复：
    1. 修正fake_cls的维度处理
    2. 重新平衡损失函数权重  
    3. 简化标签处理逻辑
    """
    assert mode in ["train_D", "train_G"]
    
    N = len(generators)

    # discriminator output for real data
    with get_autocast_context(args.amp_dtype):
        # 真实数据通过判别器
        dis_real_outputs = [model(y, label) for (model, y, label) in zip(discriminators, Y, LABELS)]
        
        # 生成器前向传播
        outputs = [generator(x) for (generator, x) in zip(generators, X)]
        real_labels = [torch.ones_like(dis_real_output).to(device) for dis_real_output in dis_real_outputs]
        
        # 分离回归输出和分类logits
        fake_data_G, fake_logits_G = zip(*outputs)
        
        # 从logits获取分类预测 - 关键修复！
        fake_cls_G = [torch.argmax(logit, dim=1) for logit in fake_logits_G]  # shape: [batch_size]
        
        # 真实数据判别损失
        lossD_real = [criterion(dis_real_output, real_label) for (dis_real_output, real_label) in
                      zip(dis_real_outputs, real_labels)]

    if mode == "train_D":
        # 训练判别器时，分离梯度
        fake_data_temp_G = [fake_data.detach() for fake_data in fake_data_G]
        fake_cls_temp_G = [fake_cls.detach() for fake_cls in fake_cls_G]
    else:
        # 训练生成器时，保持梯度
        fake_data_temp_G = fake_data_G
        fake_cls_temp_G = fake_cls_G

    # 构造时间序列数据 - 修复标签维度问题
    fake_data_sequences = []
    fake_cls_sequences = []
    
    for i, (y, label, window_size, fake_data, fake_cls) in enumerate(
        zip(Y, LABELS, window_sizes, fake_data_temp_G, fake_cls_temp_G)):
        
        # 回归数据：拼接历史数据和生成的新数据
        fake_seq = torch.cat([y[:, :window_size, :], fake_data.reshape(-1, 1, target_num)], dim=1)
        fake_data_sequences.append(fake_seq)
        
        # 分类标签：拼接历史标签和生成的新标签 - 关键修复！
        # fake_cls应该是整数标签，不需要target_num维度
        fake_cls_expanded = fake_cls.unsqueeze(1).unsqueeze(2)  # [batch_size] -> [batch_size, 1, 1]
        fake_cls_seq = torch.cat([label[:, :window_size, :], fake_cls_expanded], dim=1)
        fake_cls_sequences.append(fake_cls_seq)

    # 处理不同窗口大小的数据对齐
    fake_data_GtoD = {}
    fake_cls_GtoD = {}
    
    for i in range(N):
        for j in range(N):
            if i < j:
                # 如果生成器i的窗口小于判别器j，需要补齐前面的数据
                gap = window_sizes[j] - window_sizes[i]
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = torch.cat(
                    [Y[j][:, :gap, :], fake_data_sequences[i]], dim=1)
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = torch.cat(
                    [LABELS[j][:, :gap, :], fake_cls_sequences[i]], dim=1)
            elif i > j:
                # 如果生成器i的窗口大于判别器j，需要截取后面的数据
                gap = window_sizes[i] - window_sizes[j]
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_sequences[i][:, gap:, :]
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_cls_sequences[i][:, gap:, :]
            else:
                # 窗口大小相同，直接使用
                fake_data_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_data_sequences[i]
                fake_cls_GtoD[f"G{i + 1}ToD{j + 1}"] = fake_cls_sequences[i]

    fake_labels = [torch.zeros_like(real_label).to(device) for real_label in real_labels]

    with get_autocast_context(args.amp_dtype):
        # 判别器对假数据的输出
        dis_fake_outputD = []
        for i in range(N):
            row = []
            for j in range(N):
                fake_data_input = fake_data_GtoD[f"G{j + 1}ToD{i + 1}"]
                fake_cls_input = fake_cls_GtoD[f"G{j + 1}ToD{i + 1}"].long()
                
                out = discriminators[i](fake_data_input, fake_cls_input)
                row.append(out)
            if mode == "train_D":
                row.append(lossD_real[i])
            dis_fake_outputD.append(row)

        # 计算损失矩阵
        if mode == "train_D":
            loss_matrix = torch.zeros(N, N + 1, device=device)
            weight = weight_matrix.clone().detach()  # [N, N+1]
            for i in range(N):
                for j in range(N + 1):
                    if j < N:
                        loss_matrix[i, j] = criterion(dis_fake_outputD[i][j], fake_labels[i])
                    elif j == N:
                        loss_matrix[i, j] = dis_fake_outputD[i][j]
        elif mode == "train_G":
            loss_matrix = torch.zeros(N, N, device=device)
            weight = weight_matrix.clone().detach()  # [N, N]
            for i in range(N):
                for j in range(N):
                    loss_matrix[i, j] = criterion(dis_fake_outputD[i][j], real_labels[i])

        # 计算总损失
        loss_DorG = torch.multiply(weight, loss_matrix).sum(dim=1)  # [N]

        if mode == "train_G":
            # 回归损失 - 使用修正的权重
            loss_mse_G = [F.mse_loss(fake_data.squeeze(), y[:, -1, :].squeeze()) for (fake_data, y) in
                          zip(fake_data_G, Y)]
            mse_losses = torch.stack(loss_mse_G).to(device)
            
            # 分类损失 - 修正标签处理
            cls_losses = []
            for fake_logit, label in zip(fake_logits_G, LABELS):
                true_label = label[:, -1, :].squeeze().long()  # 获取最后一个时间步的真实标签
                cls_loss = F.cross_entropy(fake_logit, true_label)
                cls_losses.append(cls_loss)
            cls_losses = torch.stack(cls_losses).to(device)
            
            # 组合损失 - 使用不同权重
            combined_losses = mse_weight * mse_losses + cls_weight * cls_losses
            loss_DorG = loss_DorG + combined_losses
            
            # 返回详细的损失信息
            loss_matrix = mse_losses  # 保持兼容性，返回MSE损失用于记录

    return loss_DorG, loss_matrix


def do_distill(rank, generators, dataloaders, optimizers, window_sizes, device,
               *,
               alpha: float = 0.3,  # 软目标权重
               temperature: float = 2.0,  # 温度系数
               grad_clip: float = 1.0,  # 梯度裁剪上限 (L2‑norm)
               mse_lambda: float = 0.8,
               ):
    teacher_generator = generators[rank[0]]  # Teacher generator is ranked first
    student_generator = generators[rank[-1]]  # Student generator is ranked last
    student_optimizer = optimizers[rank[-1]]
    teacher_generator.eval()
    student_generator.train()
    # term of teacher is longer
    if window_sizes[rank[0]] > window_sizes[rank[-1]]:
        distill_dataloader = dataloaders[rank[0]]
    else:
        distill_dataloader = dataloaders[rank[-1]]
    gap = window_sizes[rank[0]] - window_sizes[rank[-1]]
    # Distillation process: Teacher generator to Student generator
    for batch_idx, (x, y, label) in enumerate(distill_dataloader):

        y = y[:, -1, :]
        y = y.to(device)
        label = label[:, -1]
        label = label.to(device)
        if gap > 0:
            x_teacher = x
            x_student = x[:, gap:, :]
        else:
            x_teacher = x[:, (-1) * gap:, :]
            x_student = x
        x_teacher = x_teacher.to(device)
        x_student = x_student.to(device)

        # Forward pass with teacher generator
        teacher_output, teacher_cls = teacher_generator(x_teacher)
        teacher_output, teacher_cls = teacher_output.detach(), teacher_cls.detach()
        # Forward pass with student generator
        student_output, student_cls = student_generator(x_student)


        # 使用温度缩放后计算 softmax 分布
        teacher_soft = F.softmax(teacher_cls.detach() / temperature, dim=1)
        student_log_soft = F.log_softmax(student_cls / temperature, dim=1)

        # 软标签学习损失：KL 散度
        soft_loss = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean") * (alpha * temperature ** 2)

        label_onehot = F.one_hot(label.long(), num_classes=student_cls.size(1)).float()

        # 硬目标损失：学生分类输出和真实标签计算交叉熵
        hard_loss = nn.BCEWithLogitsLoss()(student_cls, label_onehot) * (1 - alpha)
        hard_loss += F.mse_loss(student_output * temperature, y) * (1 - alpha) * mse_lambda
        distillation_loss = soft_loss + hard_loss

        # Backpropagate the loss and update student generator
        student_optimizer.zero_grad()
        # distillation_loss.backward()
        scaler.scale(distillation_loss).backward()

        if grad_clip is not None:
            clip_grad_norm_(student_generator.parameters(), grad_clip)

        # student_optimizer.step()  # Assuming same optimizer for all generators, modify as needed
        scaler.step(student_optimizer)
        scaler.update()


if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path

    # 将当前文件所在目录的上级加入 sys.path
    sys.path.append(str(Path(__file__).resolve().parent.parent))