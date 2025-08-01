# 结合生存分析的中医智能处方推荐系统

- ## datasets

  包含所有数据文件

- ## mode

  包含MLKNN、KNN训练模型

- ## static

  > ### **cluster**
  >
  > > 人群划分结果数据
  >
  > ### **CSS**
  >
  > > 全部CSS代码
  >
  > ### **images**
  >
  > > 图片资源
  >
  > ### **js**
  >
  > > JavaScript文件
  >
  > ### **uploads**
  >
  > > 用户上传的数据集文件

- ## templates

  **全部前端代码**

  - **base.html**

    基模板，导航栏功能

  - **dataset.html**

    数据集管理前端

  - **home.html**

    平台首页

  - **index.html**

    欢迎界面

  - **input.html**

    问诊信息采集

  - **log.html**

    诊断日志

  - **login.html**

    登录界面

  - **model.html**

    模型管理

  - **other_home.html**

    他人主页

  - **output.html**

    推荐结果

  - **people_divide.html**

    人群划分

  - **people_divide_info.html**

    人群详细信息

  - **report.html**

    诊断报告

  - **signup.html**

    登录界面

  - **user_home.html**

    用户主页

- ## python文件

  - **algorithm.py**

    mlknn、knn

  - **Clustering2Analysis.py**

    人群划分

  - **data_check.py**

    人群划分数据格式检测

  - **jaccard.py**

    计算相似人群

  - **main.py**

    flask后端

  - **PR_system_sim.py**

    相似度推荐算法

  - **similarity.py**

    相似经典方

  