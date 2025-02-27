from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import pickle
import os
from app.models import CarFeatures, PredictionResponse
from app.prediction import predict_car_price, get_valid_feature_values
from app.utils import logger

# 加载模型组件
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                          'car_price_prediction_components.pkl')
with open(model_path, 'rb') as f:
    components = pickle.load(f)

# 创建FastAPI应用
app = FastAPI(
    title="汽车价格预测API",
    description="基于汽车特征预测价格的API",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境中应设置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建静态文件目录
static_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(static_folder, exist_ok=True)

# 挂载静态文件服务
app.mount("/static", StaticFiles(directory=static_folder), name="static")

@app.get("/")
async def root():
    """提供前端页面"""
    index_path = os.path.join(static_folder, "index.html")
    return FileResponse(index_path)

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(car: CarFeatures):
    """根据汽车特征预测价格"""
    try:
        result = predict_car_price(car, components)
        return result
    except Exception as e:
        logger.error(f"预测过程发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"预测错误: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

@app.get("/model-info")
async def model_info():
    """获取模型信息"""
    model_type = type(components['model']).__name__
    return {
        "model_type": model_type,
        "features": {
            "numerical": components['numerical_cols'],
            "categorical": components['categorical_cols']
        },
        "feature_selection": "enabled" if components.get('selected_indices') is not None else "disabled",
        "version": "1.0.0"
    }

@app.get("/feature-values")
async def feature_values():
    """获取各特征的有效值列表"""
    try:
        values = get_valid_feature_values(components)
        return values
    except Exception as e:
        logger.error(f"获取特征值时发生错误: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取特征值错误: {str(e)}")

if __name__ == "__main__":
    # 获取端口号，优先使用环境变量
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
