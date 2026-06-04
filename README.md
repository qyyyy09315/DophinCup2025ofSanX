# 2025海豚杯：三夏团队技术方案
本项目是2025年全国海豚杯大数据竞赛三夏团队的参赛作品，我们构建了融合预训练语言模型与集成学习的高性能多模态文本分类框架，在官方测试集上取得了优异的性能表现。

## 项目结构
```
DophinCup2025ofSanX/
├── data/                          # 数据集存储目录
│   ├── raw/                       # 原始未修改的竞赛数据集
│   └── processed/                 # 清洗后的数据集、BERT向量、中间输出结果
├── src/                           # 所有源代码
│   ├── preprocessing/             # 数据清洗和预处理脚本
│   │   ├── text_extraction.py     # 中文文本提取与归一化
│   │   ├── label_detection.py     # 标签校验与错误修正
│   │   ├── clean_data.py          # 主数据清洗流程
│   │   ├── test_clean.py          # 测试集数据清洗流程
│   │   └── synthetic_data_generator.py  # 用于数据增强的合成数据生成工具
│   ├── feature_engineering/       # 特征提取代码
│   │   ├── bert_embedding/        # 基于BERT的文本特征提取
│   │   │   ├── bert_convert_base.py  # 基础版BERT向量生成
│   │   │   ├── bert_convert_improved.py  # 优化版BERT向量生成
│   │   │   ├── bert_convert_efficient.py  # 高效批量BERT向量生成
│   │   │   ├── model_after_bert.py  # BERT顶层分类头
│   │   │   └── bert_xgboost_pipeline.py  # 端到端BERT + XGBoost工作流
│   │   └── utils/
│   │       └── to_xlsx.py         # 结果导出为Excel格式工具
│   ├── models/                    # 模型实现
│   │   ├── base_model_template.py  # 抽象基模型模板
│   │   ├── base_models/           # 单模型实现
│   │   │   ├── xgboost/           # 6+种XGBoost变体实现
│   │   │   ├── lightgbm/          # LightGBM实现
│   │   │   ├── catboost/          # CatBoost实现
│   │   │   ├── random_forest/     # 随机森林实现
│   │   │   └── svm/               # 支持向量机实现
│   │   ├── deep_learning/         # 神经网络模型
│   │   │   ├── dnn/               # 全连接神经网络实现
│   │   │   └── attention_dnn/     # 基于注意力机制的DNN实现
│   │   └── ensemble/              # 集成学习方法
│   │       ├── adaboost/          # 10+种AdaBoost变体实现
│   │       └── stacking/          # 融合XGBoost、CatBoost、随机森林的Stacking集成实现
│   ├── evaluation/                # 模型评估与可视化
│   │   ├── model_comparison/      # 模型性能对比脚本
│   │   ├── plotting/              # 论文图表生成脚本
│   │   └── paper_baseline_model.py  # 论文对比用基线模型
│   └── utils/                     # 共享工具库
│       ├── model_io.py            # 模型序列化/反序列化（pkl文件处理）
│       ├── inference_wrapper.py   # 通用模型推理封装
│       ├── bert_inference.py      # BERT模型推理封装
│       ├── external_training_wrapper.py  # 外部训练任务封装
│       ├── debug_tools.py         # 调试工具
│       ├── debug_config.py        # 调试配置
│       └── test_script.py         # 测试/Hello World脚本
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt               # 项目依赖列表
```

## 安装说明
1. 克隆本仓库
2. 安装所需依赖：
```bash
pip install -r requirements.txt
```

依赖包括：
- torch >= 2.0（用于BERT模型）
- transformers >= 4.30
- xgboost >= 2.0
- lightgbm >= 4.0
- catboost >= 1.2
- scikit-learn >= 1.3
- pandas >= 2.1
- numpy >= 1.25
- matplotlib >= 3.7
- seaborn >= 0.12
- pyarrow >= 14.0（支持parquet格式）

## 使用方法
### 1. 数据预处理
将原始竞赛数据集放置在`data/raw/`目录下，然后运行：
```bash
python src/preprocessing/clean_data.py
```
清洗后的数据将保存到`data/processed/`目录

### 2. 特征提取
为清洗后的文本生成BERT向量：
```bash
python src/feature_engineering/bert_embedding/bert_convert_efficient.py
```
生成的向量将保存到`data/processed/`目录

### 3. 模型训练
运行对应脚本训练任意模型，例如：
```bash
# 训练XGBoost基础模型
python src/models/base_models/xgboost/wtj_xgboost.py

# 训练AdaBoost集成模型
python src/models/ensemble/adaboost/gc_weighted_adaboost.py

# 训练Stacking集成模型（性能最优）
python src/models/ensemble/stacking/st_brf_xgboost_catboost_stacking.py
```

### 4. 模型推理
使用训练好的模型对测试数据进行预测：
```bash
python src/utils/bert_inference.py
```
结果将保存到脚本中配置的路径

### 5. 评估与可视化
生成模型对比报告和论文图表：
```bash
# 生成模型性能对比
python src/evaluation/model_comparison/comparison_set1.py

# 生成论文图表
python src/evaluation/plotting/figure1.py
```

## 核心方法
### 文本表示
我们使用针对中文优化的BERT（双向编码器表示）预训练语言模型作为特征提取器，将非结构化文本转换为768维语义向量，有效捕捉文本的上下文语义信息和长距离依赖关系。

### 模型框架
我们构建了多模型集成框架，融合多种算法的优势：
- **树类集成模型**：XGBoost、LightGBM、CatBoost、随机森林，实现高效的特征选择和非线性拟合
- **深度学习模型**：DNN和基于注意力的DNN，从语义向量中提取高阶隐含特征
- **集成策略**：使用AdaBoost对基础模型进行加权融合，同时引入Stacking集成实现多模型决策层信息互补，显著提升模型的泛化能力和鲁棒性
- **基线模型**：使用SVM作为性能对比的基线

### 性能优化
我们实现了混合精度训练、批量数据预处理和模型蒸馏，在保持模型性能的同时大幅降低训练时间。

## 致谢
本项目在开发过程中得到了多位同学和老师的大力支持：
- 感谢Z. Xia、M. Yu、Z. Liu、H. Tao同学在数据标注、模型调试、实验验证等环节提供的支持与帮助
- 特别感谢Chao Li教授在项目选题、技术路线设计、学术方法指导等方面提供的宝贵建议与帮助
