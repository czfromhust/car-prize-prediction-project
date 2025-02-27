import pandas as pd
import numpy as np
from app.models import CarFeatures, PredictionResponse
from app.utils import logger

def predict_car_price(car: CarFeatures, components: dict) -> PredictionResponse:
    # 转换输入数据为DataFrame
    input_data = car.dict()
    
    # 处理特定字段名称不一致的问题
    if 'Prod_year' in input_data and 'Prod. year' in components['model_features']:
        input_data['Prod. year'] = input_data.pop('Prod_year')
    
    input_df = pd.DataFrame([input_data])
    
    # 转换列名，去掉下划线
    input_df.columns = [col.replace('_', ' ') for col in input_df.columns]
    
    # 记录输入数据用于调试
    logger.info(f"输入数据: {input_df.to_dict()}")

    # 处理数值特征的缺失值
    for col in components['numerical_cols']:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            if pd.isna(input_df[col]).any():
                input_df[col].fillna(components['numerical_impute_values'][col], inplace=True)

    # 处理分类特征的缺失值和标准化处理
    for col in components['categorical_cols']:
        if col in input_df.columns:
            # 确保数据是字符串类型
            input_df[col] = input_df[col].astype(str)
            
            # 统一转换为小写并去除首尾空格，与训练时处理保持一致
            input_df[col] = input_df[col].str.lower().str.strip()
            
            # 处理缺失值
            if pd.isna(input_df[col]).any():
                input_df[col].fillna(components['categorical_impute_values'][col], inplace=True)
            
            logger.info(f"处理后的 {col}: {input_df[col].tolist()}")

    # 应用标签编码 - 修复的关键部分
    for col in components['categorical_cols']:
        if col in input_df.columns:
            encoder = components['label_encoders'][col]
            
            # 找到训练数据中最常见的类别用作替代值
            default_value = encoder.classes_[0] if len(encoder.classes_) > 0 else "unknown"
            
            # 创建一个新的列，确保值在编码器的类别中
            safe_values = []
            for idx, val in enumerate(input_df[col]):
                if val not in encoder.classes_:
                    logger.warning(f"特征 {col} 的值 '{val}' 不在训练数据中，使用 '{default_value}' 替代")
                    safe_values.append(default_value)
                else:
                    safe_values.append(val)
            
            # 使用安全值替换原始列
            input_df[col] = safe_values
            
            try:
                # 应用编码器转换
                input_df[col] = encoder.transform(input_df[col])
            except ValueError as e:
                logger.error(f"编码错误在特征 {col} 上: {str(e)}")
                raise ValueError(f"特征 {col} 包含未知标签: {input_df[col].tolist()}")

    # 应用数值特征缩放
    cols_to_scale = [col for col in components['numerical_cols'] if col in input_df.columns and col != 'Price']
    if cols_to_scale:
        input_df[cols_to_scale] = components['scaler'].transform(input_df[cols_to_scale])

    # 确保所有需要的特征都存在
    missing_features = [feat for feat in components['model_features'] if feat not in input_df.columns]
    if missing_features:
        logger.error(f"缺少以下特征: {missing_features}")
        raise ValueError(f"输入数据缺少以下必要特征: {missing_features}")

    # 准备特征，确保特征顺序与训练时一致
    features = input_df[components['model_features']]
    
    # 应用特征选择（如果有）
    if components.get('selected_indices') is not None:
        features = features.iloc[:, components['selected_indices']]

    # 预测价格
    predicted_price = components['model'].predict(features)[0]
    logger.info(f"预测价格: {predicted_price}")

    # 计算置信区间（如果模型支持）
    confidence_interval = None
    if hasattr(components['model'], 'estimators_'):
        predictions = [tree.predict(features)[0] for tree in components['model'].estimators_]
        confidence_interval = {
            "lower_bound": float(np.percentile(predictions, 2.5)),
            "upper_bound": float(np.percentile(predictions, 97.5))
        }

    # 计算特征重要性（如果模型支持）
    feature_importance = None
    if hasattr(components['model'], 'feature_importances_'):
        importances = components['model'].feature_importances_
        feature_names = features.columns
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, importances)}
        feature_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5])

    return {
        "predicted_price": float(predicted_price),
        "confidence_interval": confidence_interval,
        "feature_importance": feature_importance
    }

# 添加一个帮助函数来获取特征有效值，用于前端
def get_valid_feature_values(components):
    """获取每个分类特征的有效值列表"""
    valid_values = {}
    for col in components['categorical_cols']:
        print(col)
        encoder = components['label_encoders'][col]
        valid_values[col] = encoder.classes_.tolist()
    return valid_values