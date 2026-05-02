# Gated-PINN-Transformer
门控物理信息神经网络——Transformer论文的源代码。基础python为3.14。
由于源数据太多，github无法上传，故上传至该地址：
先使用预处理程序data_preprocessing_plastic_data.py处理数据。 cycle_ratio为前n%数据，可更改； scale_range为数据增强的比例，可更改。
config中的超参数可自由调整以适应不同比例的训练集数据。
超参数不变，random_seed不变，则不同批次运行结果一致，保证可复现。改变超参数或random_seed会导致不同批次运行结果不同。
