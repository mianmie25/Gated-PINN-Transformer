# Gated-PINN-Transformer
门控物理信息神经网络——Transformer论文的源代码。基础python为3.14。
先使用预处理程序data_preprocessing_plastic_data.py处理数据。 cycle_ratio为前n%数据，可更改； scale_range为数据增强的比例，可更改。
config中的超参数可自由调整以适应不同比例的训练集数据。
超参数不变，random_seed不变，则不同批次运行结果一致，可复现。改变超参数或random_seed会导致不同批次运行结果不同。
程序中的材料相关参数换成自己的。
依次删去gated-pinn-transformer中的gated模块和pinn模块即可得gated-transformer与pinn-transformer。
