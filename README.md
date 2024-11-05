代码分为训练和推理测试两部分
1.环境配置
在 Python>=3.8 环境中使用 PyTorch>=1.8 通过 pip 安装包含所有依赖项 的 ultralytics 包。
pip install ultralytics
2.推理测试
##推理测试页面,yolo文件夹内运行
streamlit run trainapp.py 
##此镜像环境包含Ultralytics库中的所有模型均可以使用
要想运行自己的模型可视化效果的话，可以将模型文件置于weights / detection路径下面即可运行页面程序，
运行如下
推理运行阶段：


上传后检测过程如图：

3.训练过程
3.1数据集准备，
首先下载teample里面的数据集结构文件夹进行上传数据集
格式如下：
train
├── images
└── labels
valid
├── images
└── labels
yolo数据集标注格式：
class_id        x                y            w           h

class_id: 类别的id编号
x: 目标的中心点x坐标(横向) /图片总宽度
y: 目标的中心的y坐标(纵向) /图片总高度
w:目标框的宽度/图片总宽度
h: 目标框的高度/图片总高度
例如：

3.2训练
文件路径：
lanyun-tmp/ultralytics/ultralytics/cfg/datasets/
内容格式可以仿照：
3.3训练
运行图中的文件即可进行训练
可以选择模型以及填写参数数据集类别等：

填写完成后上传提前制作好的数据集压缩包一键运行训练。
训练过程展示：

训练完成后日志展示：



3.4结果查看：
路径：
lanyun-tmp/yolo/runs/detect/

