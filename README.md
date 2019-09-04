![1](https://raw.githubusercontent.com/DASTUDIO/babycare/master/img/1.png)

![1](https://raw.githubusercontent.com/DASTUDIO/babycare/master/img/2.png)

干净简单的DNN文本识别。 [示例](http://babycare.da.studio)

* 依赖

```python
pip3 install numpy
pip3 install jieba
pip3 install tensorflow
pip3 install keras
```

>坏训练信息放在`data_set/bad_data_train`一行一个
>
>好训练信息放在`data_set/good_data_train`一行一个

* 训练

```python
python3 care/dnn/train.py
```

* 运行web

```
python3 manage.py runserver 0.0.0.0:80
```

* web 接口

```
url?data=yourStrings->{yourStrings, res} # res越接近1越可能是敏感信息
```

>数据库配置care/view.py 按需修改
>
>网络结构care/dnn/isBad.py 按需修改


