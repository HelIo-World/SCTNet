from .builder import DATASETS
from . import BaseSegDataset

@DATASETS.register_module()
class MarsSegDataset(BaseSegDataset):
    METAINFO = dict(
        # 一共 10 个有效类别（索引 0 到 9）
        classes=('soil', 'sand', 'gravel', 'bedrock', 'rock', 
                 'track', 'shadow', 'background', 'unknown'),
        # 为可视化配置的 RGB 颜色表
        palette=[
            (128, 0, 0),         
            (0, 128, 0),   
            (128, 128, 0),     
            (0, 0, 128),    
            (128, 0, 128),    
            (0, 128, 128),   
            (128, 128, 128),   
            (192, 0, 0),     
            (64, 0, 0)   
        ]
    )
