# 汽车价格预测系统

这是一个基于机器学习的汽车价格预测系统，使用随机森林算法根据车辆特征预测二手车价格。该项目包含完整的数据分析、模型训练和Web应用部署流程。

## 项目概述

本项目通过分析汽车的各种特征（如制造商、型号、里程数、发动机容量等）来预测汽车的价格。整个系统包括：

1. 数据分析与预处理
2. 模型训练与优化
3. FastAPI后端API服务
4. 交互式Web前端界面

## 技术栈

- **数据分析**: Pandas, NumPy, Matplotlib, Seaborn
- **机器学习**: Scikit-learn, 随机森林回归
- **后端**: FastAPI, Uvicorn
- **前端**: HTML, CSS, JavaScript, Bootstrap, Select2
- **开发工具**: Jupyter Notebook, Python

## 项目结构

```
汽车价格预测系统/
├── app/                    # FastAPI应用
│   ├── __init__.py        
│   ├── main.py             # FastAPI主程序
│   ├── models.py           # 数据模型定义
│   ├── prediction.py       # 预测逻辑
│   └── utils.py            # 工具函数
├── static/                 # 静态文件
│   └── index.html          # 前端界面
├── logs/                   # 日志文件
├── real-car-predict.ipynb  # 模型训练与分析Jupyter笔记本
├── car_price_prediction_components.pkl  # 保存的模型和处理组件
└── README.md               # 项目说明文档
```

## 特征说明

本模型考虑了以下特征：

### 分类特征
- Manufacturer (制造商)
- Model (型号)
- Category (类别)
- Leather interior (皮革内饰)
- Fuel type (燃料类型)
- Gear box type (变速箱类型)
- Drive wheels (驱动轮)
- Doors (车门数)
- Wheel (方向盘位置)
- Color (颜色)

### 数值特征
- Levy (征税额)
- Prod. year (生产年份)
- Engine volume (发动机容量)
- Cylinders (气缸数)
- Airbags (安全气囊数)
- Mileage (里程数)

## 模型开发过程

### 1. 数据分析
- 探索性数据分析（EDA）
- 特征分布可视化
- 相关性分析
- 缺失值处理

### 2. 模型选择
通过比较多种回归算法（线性回归、随机森林、梯度提升等），最终选择了随机森林模型，原因包括：
- 能够捕捉特征间的非线性关系
- 对异常值和噪声具有良好的鲁棒性
- 提供特征重要性分析
- 在验证集上表现最佳

### 3. 超参数调优
采用分阶段网格搜索策略优化模型：
1. 基线模型评估
2. 树的复杂度参数调优（max_depth, min_samples_split等）
3. 森林大小和随机性调优（n_estimators, max_features等）
4. 整合最佳参数并添加正则化
5. 特征重要性分析与筛选

### 4. 模型评估
- 使用R²和MSE作为评估指标
- 交叉验证确保模型稳定性
- 分析训练集和验证集表现差异，避免过拟合

## API接口说明

### 主要端点
- `GET /` - 提供前端界面
- `POST /predict` - 根据汽车特征预测价格
- `GET /feature-values` - 获取分类特征的有效值列表
- `GET /health` - 健康检查端点
- `GET /model-info` - 获取模型信息

### 示例请求（预测）
```json
{
  "Manufacturer": "TOYOTA",
  "Model": "Camry",
  "Category": "Sedan",
  "Leather_interior": "Yes",
  "Fuel_type": "Petrol",
  "Gear_box_type": "Automatic",
  "Drive_wheels": "Front",
  "Doors": "04-May",
  "Wheel": "Left wheel",
  "Color": "Black",
  "Levy": 1200,
  "Prod_year": 2018,
  "Engine_volume": 2.5,
  "Cylinders": 4,
  "Airbags": 8,
  "Mileage": 25000
}
```

### 示例响应
```json
{
  "predicted_price": 15800.0,
  "confidence_interval": {
    "lower_bound": 14200.0,
    "upper_bound": 17400.0
  },
  "feature_importance": {
    "Prod_year": 0.3421,
    "Engine_volume": 0.1843,
    "Mileage": 0.1276,
    "Manufacturer": 0.0982,
    "Model": 0.0731
  }
}
```

## 前端界面

前端界面提供了用户友好的表单，用于输入汽车特征并获取价格预测：

- 所有分类特征使用下拉选择框
- 当选项超过10个时自动启用搜索功能
- 美观的结果展示，包括预测价格、置信区间和特征重要性
- 自适应设计，支持各种设备

## 部署说明

### 环境要求
- Python 3.8+
- 依赖包：fastapi, uvicorn, scikit-learn, pandas, numpy

### 安装步骤
1. 克隆或下载代码库
2. 安装依赖：`pip install -r requirements.txt`
3. 运行服务器：`python -m app.main`
4. 在浏览器访问：`http://localhost:8000`

## 模型优势

随机森林在本项目中表现优异的原因：

1. **非线性关系捕捉**：汽车价格与特征（如里程数、发动机容量等）之间往往是非线性关系
2. **鲁棒性强**：随机森林作为集成方法，对噪声和异常值有良好的抵抗力
3. **特征重要性**：自然支持特征重要性分析，帮助理解哪些变量对价格预测贡献最大
4. **适应高维数据**：能够有效处理多个数值和类别特征

## 未来改进方向

1. 增加更多特征工程技术
2. 尝试更多高级模型（如XGBoost、LightGBM）
3. 添加模型解释组件（SHAP值分析）
4. 优化前端用户体验
5. 实现模型定期重训练机制

## 贡献

欢迎贡献代码或提供改进建议！

## 许可

MIT