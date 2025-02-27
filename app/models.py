from pydantic import BaseModel, Field
from typing import Dict, Optional, Union, List

class CarFeatures(BaseModel):
    """汽车特征输入模型"""
    Manufacturer: str = Field(..., description="制造商")
    Model: str = Field(..., description="型号")
    Category: str = Field(..., description="类别")
    Leather_interior: str = Field(..., description="皮革内饰 (Yes/No)")
    Fuel_type: str = Field(..., description="燃料类型")
    Gear_box_type: str = Field(..., description="变速箱类型")
    Drive_wheels: str = Field(..., description="驱动轮")
    Doors: str = Field(..., description="车门数")  # 改为字符串类型，因为原始数据包含"04-May"等格式
    Wheel: str = Field(..., description="方向盘位置 (Left/Right)")
    Color: str = Field(..., description="颜色")
    Levy: Optional[float] = Field(None, description="征税额")
    Prod_year: Optional[int] = Field(None, description="生产年份")  # 在预测逻辑中会转为 'Prod. year'
    Engine_volume: Optional[float] = Field(None, description="发动机容量")
    Cylinders: Optional[int] = Field(None, description="气缸数")
    Airbags: Optional[int] = Field(None, description="安全气囊数")
    Mileage: Optional[float] = Field(None, description="里程数")
    
    class Config:
        schema_extra = {
            "example": {
                "Manufacturer": "TOYOTA",
                "Model": "Camry",
                "Category": "Sedan",
                "Leather_interior": "Yes",
                "Fuel_type": "Petrol",
                "Gear_box_type": "Automatic",
                "Drive_wheels": "Front",
                "Doors": "04-May",  # 使用与训练数据一致的格式
                "Wheel": "Left wheel",  # 使用与训练数据一致的格式
                "Color": "Black",
                "Levy": 1200,
                "Prod_year": 2018,
                "Engine_volume": 2.5,
                "Cylinders": 4,
                "Airbags": 8,
                "Mileage": 25000
            }
        }

class PredictionResponse(BaseModel):
    """预测响应模型"""
    predicted_price: float = Field(..., description="预测价格")
    confidence_interval: Optional[Dict[str, float]] = Field(None, description="预测置信区间")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="特征重要性")