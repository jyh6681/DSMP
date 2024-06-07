import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# 数据
layers = [1, 2,3,4,5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
model_names = ["DSMP","GCN","GAT"]
color_codes = ['#0071bc', '#00a170', '#e51633', '#9e008e', '#ffc72c', '#58595b']#['#E8A628', '#B87D4B', '#6B4226']
# 创建一个图形对象
fig, ax = plt.subplots()
for model_name in model_names:
    if model_name=="DSMP":
        globals()[model_name+"_mean_accuracy"] = [0.75, 0.7589, 0.7959, 0.7827, 0.7619, 0.756, 0.7738, 0.7649, 0.7768, 0.7679, 0.7619, 0.747, 0.756, 0.7768]
        globals()[model_name+"_variance"] =  [0.0062, 0.0076, 0.0074, 0.0058, 0.0047, 0.002, 0.0044, 0.0024, 0.0089, 0.0054, 0.0028, 0.0131, 0.0049, 0.007]
        # 设置RGB颜色
        color = (0.2, 0.4, 0.6)  # 深蓝色
        colora = (0.2, 0.4, 0.6, 0.6)  # 半透明的深蓝色
    if model_name=="GCN":
        globals()[model_name+"_mean_accuracy"] = [0.3455, 0.705, 0.709, 0.704, 0.705, 0.688, 0.702, 0.695, 0.703, 0.694, 0.694, 0.682, 0.70, 0.692] 
        globals()[model_name+"_variance"] =   [0.0591, 0.0128, 0.013, 0.002, 0.0055, 0.0033, 0.0013, 0.001, 0.0016, 0.004, 0.0028, 0.0016, 0.0018, 0.0016]
        # 设置RGB颜色
        color = (210/255, 180/255, 140/255)
        colora = (210/255, 180/255, 140/255, 0.6)  
    if model_name=="GAT":
        globals()[model_name+"_mean_accuracy"] =[0.3339, 0.695, 0.702, 0.701, 0.703, 0.698, 0.688, 0.702, 0.698, 0.676, 0.692, 0.688, 0.6857, 0.679] 
        globals()[model_name+"_variance"] =   [0.0915, 0.008, 0.007, 0.0121, 0.0091, 0.002, 0.0008, 0.0015, 0.0015, 0.0005, 0.0041, 0.0019, 0.0021, 0.0006]
        # 设置RGB颜色
        color = (139/255, 0, 0)
        colora = (139/255, 0, 0, 0.6)  
    #  计算上下界
    lower_bound = np.array(globals()[model_name+"_mean_accuracy"]) - np.array(globals()[model_name+"_variance"])
    upper_bound = np.array(globals()[model_name+"_mean_accuracy"]) + np.array(globals()[model_name+"_variance"])

    # 绘制曲线图
    # ax.plot(layers, globals()[model_name+"_mean_accuracy"], color=color, linewidth=2, label=model_name)
    sns.lineplot(x=layers, y=globals()[model_name+"_mean_accuracy"],label=model_name,color=color)
    ax.fill_between(layers, lower_bound, upper_bound, color=colora, alpha=0.4)

    # 设置图表标题和轴标签
    ax.set_xlabel('Layers')
    ax.set_ylabel('Accuracy')

    # 添加图例
    ax.legend()
# ax.set_title('Accuracy vs Layers')
# 显示图表
plt.show()
plt.savefig('/home/jyh_temp1/Downloads/scatter_MP/ScatteringMP/show_result/depth_vs_acc_sns2.png')

# if model_name=="GCN":
#     globals()[model_name+"_mean_accuracy"] = [0.3455, 0.7179, 0.7505, 0.7409, 0.7452, 0.6884, 0.7125, 0.7214, 0.7036, 0.6946, 0.6946, 0.6821, 0.7, 0.692] 
#     globals()[model_name+"_variance"] =   [0.0591, 0.0168, 0.0029, 0.002, 0.0055, 0.0033, 0.0013, 0.001, 0.0016, 0.004, 0.0028, 0.0016, 0.0018, 0.0016]
#     # 设置RGB颜色
#     color = (210/255, 180/255, 140/255)
#     colora = (210/255, 180/255, 140/255, 0.6)  
# if model_name=="GAT":
#     globals()[model_name+"_mean_accuracy"] =[0.3339, 0.7259, 0.7791, 0.7543, 0.73045, 0.708, 0.7304, 0.7089, 0.6982, 0.6768, 0.6929, 0.6884, 0.6857, 0.6795] 
#     globals()[model_name+"_variance"] =   [0.0915, 0.008, 0.007, 0.0121, 0.0091, 0.002, 0.0008, 0.0015, 0.0015, 0.0005, 0.0041, 0.0019, 0.0021, 0.0006]
        