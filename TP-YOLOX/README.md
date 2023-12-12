## TP-YOLOX：复杂环境下的茶树害虫检测
---

## 所需环境
python==3.7 pytorch==1.7.1

 
## 预测
1. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
2. 运行predict.py，输入  
```python
img/street.jpg
```
3. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

## 评估
运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

## Reference
https://github.com/Megvii-BaseDetection/YOLOX
其它细节待投稿成功后再进行补充，如有急需可以直接找原作者要实验数据.