import logging
import os

def setup_logger():
    """设置日志记录器"""
    logger = logging.getLogger("car_price_api")
    
    # 如果已经有处理器，不再添加
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 可选：添加文件处理器以记录到日志文件
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'car_price_api.log'))
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# 创建全局日志记录器实例
logger = setup_logger()