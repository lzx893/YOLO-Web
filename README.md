代码分为训练和推理测试两部分

# 1.环境配置
<font style="color:rgb(31, 35, 40);">在 </font>[Python>=3.8](https://www.python.org/)<font style="color:rgb(31, 35, 40);"> 环境中使用 </font>[PyTorch>=1.8](https://pytorch.org/get-started/locally/)<font style="color:rgb(31, 35, 40);"> 通过 pip 安装包含所有</font>[依赖项](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml)<font style="color:rgb(31, 35, 40);"> 的 ultralytics 包。</font>

<font style="color:rgb(31, 35, 40);">pip install ultralytics</font>

# 2.推理测试
##推理测试页面,yolo文件夹内运行

streamlit run trainapp.py 

##此镜像环境包含Ultralytics库中的所有模型均可以使用

要想运行自己的模型可视化效果的话，可以将模型文件置于weights / detection路径下面即可运行页面程序，

运行如下

推理运行阶段：

![](https://cdn.nlark.com/yuque/0/2024/png/42455527/1728871795152-9609d29b-d3a8-4536-95b8-74a13b2aa747.png)

![](https://cdn.nlark.com/yuque/0/2024/png/42455527/1728871781096-3fe4a22e-6db2-446a-8a5b-29ba146791d9.png)

上传后检测过程如图：

![](https://cdn.nlark.com/yuque/0/2024/png/42455527/1728871915245-e6795c56-1190-40ed-bc7d-914ce02d1d88.png)

# 3.训练过程
## 3.1数据集准备，
首先下载teample里面的数据集结构文件夹进行上传数据集

格式如下：

<font style="color:rgb(79, 79, 79);">train  
</font><font style="color:rgb(79, 79, 79);">├── images  
</font><font style="color:rgb(79, 79, 79);">└── labels</font>

<font style="color:rgb(79, 79, 79);">valid  
</font><font style="color:rgb(79, 79, 79);">├── images  
</font><font style="color:rgb(79, 79, 79);">└── labels</font>

yolo数据集标注格式：

<font style="color:rgb(79, 79, 79);background-color:rgb(238, 240, 244);">class_id        x                y            w           h</font>

<font style="color:rgb(79, 79, 79);background-color:rgb(238, 240, 244);"></font>

<font style="color:rgb(79, 79, 79);background-color:rgb(238, 240, 244);">class_id: 类别的id编号  
</font><font style="color:rgb(79, 79, 79);background-color:rgb(238, 240, 244);">x: 目标的中心点x坐标(横向) /图片总宽度  
</font><font style="color:rgb(79, 79, 79);background-color:rgb(238, 240, 244);">y: 目标的中心的y坐标(纵向) /图片总高度  
</font><font style="color:rgb(79, 79, 79);background-color:rgb(238, 240, 244);">w:目标框的宽度/图片总宽度  
</font><font style="color:rgb(79, 79, 79);background-color:rgb(238, 240, 244);">h: 目标框的高度/图片总高度</font>

<font style="color:rgb(79, 79, 79);background-color:rgb(238, 240, 244);">例如：</font>

![](https://cdn.nlark.com/yuque/0/2024/png/42455527/1728872233101-d847f10a-6737-4955-a7b7-d2f6410445d0.png)

## 3.2训练
文件路径：

lanyun-tmp/ultralytics/ultralytics/cfg/datasets/

内容格式可以仿照：

## 3.3训练
运行图中的文件即可进行训练

可以选择模型以及填写参数数据集类别等：

![](https://cdn.nlark.com/yuque/0/2024/png/42455527/1729058361121-2cf5bba7-220b-4681-b2d7-d507ed973779.png)

填写完成后上传提前制作好的数据集压缩包一键运行训练。

训练过程展示：

![](https://cdn.nlark.com/yuque/0/2024/png/42455527/1729058471800-e52b6430-3818-475a-991e-eaec3c783cc7.png)

训练完成后日志展示：

![](https://cdn.nlark.com/yuque/0/2024/png/42455527/1729058620937-f06a1eb5-a522-4961-9f1f-d86c1026bff3.png)





## 3.4结果查看：
路径：

lanyun-tmp/yolo/runs/detect/

![](https://cdn.nlark.com/yuque/0/2024/png/42455527/1728872799431-c0958850-f12e-4e6b-b3db-70a80068cafc.png)



