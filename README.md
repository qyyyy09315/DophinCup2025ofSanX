
<div align="center">
  <h1>🐬 2025海豚杯：三夏团队技术方案</h1>
  <p>
    <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python Version">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch Version">
    <img src="https://img.shields.io/badge/Transformer-4.30+-ff6f61?logo=huggingface" alt="Transformer Version">
    <img src="https://img.shields.io/badge/模型精度-93.7%25-green" alt="Model Accuracy">
  </p>
  <p>
    本项目为2025年全国海豚杯大数据竞赛三夏团队参赛作品，构建了融合预训练语言模型与集成学习的高性能多模态文本分类框架，在官方测试集上取得Top3的优异成绩。
  </p>
</div>

---

## 📋 项目概览
> 🎯 **核心目标**：针对竞赛提供的多领域中文文本数据，实现高精度多标签分类，解决数据分布不均、文本噪声大、领域差异明显等挑战。
>
> ✨ **核心优势**：
> - 采用BERT预训练模型提取深度语义特征，解决文本语义理解难题
> - 多模型集成框架融合树模型、深度学习模型优势，泛化能力强
> - 全流程工程化实现，支持快速迭代和部署

---

## 📂 项目结构
```
DophinCup2025ofSanX/
├── 📁 data/                          # 数据集存储目录
│   ├── 📁 raw/                       # 原始未修改的竞赛数据集（请勿修改）
│   └── 📁 processed/                 # 清洗后的数据集、BERT向量、中间输出结果
├── 📁 src/                           # 所有源代码
│   ├── 📁 preprocessing/             # 数据清洗和预处理脚本
│   │   ├── text_extraction.py        # 中文文本提取与归一化
│   │   ├── label_detection.py        # 标签校验与错误修正
│   │   ├── clean_data.py             # 主数据清洗流程
│   │   ├── test_clean.py             # 测试集数据清洗流程
│   │   └── synthetic_data_generator.py # 数据增强合成工具
│   ├── 📁 feature_engineering/       # 特征提取代码
│   │   ├── 📁 bert_embedding/        # 基于BERT的文本特征提取
│   │   │   ├── bert_convert_base.py  # 基础版BERT向量生成
│   │   │   ├── bert_convert_improved.py # 优化版BERT向量生成
│   │   │   ├── bert_convert_efficient.py # 高效批量BERT向量生成
│   │   │   ├── model_after_bert.py   # BERT顶层分类头
│   │   │   └── bert_xgboost_pipeline.py # 端到端BERT+XGBoost工作流
│   │   └── 📁 utils/
│   │       └── to_xlsx.py            # 结果导出为Excel格式工具
│   ├── 📁 models/                    # 模型实现
│   │   ├── base_model_template.py    # 抽象基模型模板
│   │   ├── 📁 base_models/           # 单模型实现
│   │   │   ├── xgboost/              # 6+种XGBoost变体实现
│   │   │   ├── lightgbm/             # LightGBM实现
│   │   │   ├── catboost/             # CatBoost实现
│   │   │   ├── random_forest/        # 随机森林实现
│   │   │   └── svm/                  # 支持向量机实现
│   │   ├── 📁 deep_learning/         # 神经网络模型
│   │   │   ├── dnn/                  # 全连接神经网络实现
│   │   │   └── attention_dnn/        # 基于注意力机制的DNN实现
│   │   └── 📁 ensemble/              # 集成学习方法
│   │       ├── adaboost/             # 10+种AdaBoost变体实现
│   │       └── stacking/             # 融合XGBoost/CatBoost/RF的Stacking集成
│   ├── 📁 evaluation/                # 模型评估与可视化
│   │   ├── model_comparison/         # 模型性能对比脚本
│   │   ├── plotting/                 # 论文图表生成脚本
│   │   └── paper_baseline_model.py   # 论文对比用基线模型
│   └── 📁 utils/                     # 共享工具库
│       ├── model_io.py               # 模型序列化/反序列化（pkl处理）
│       ├── inference_wrapper.py      # 通用模型推理封装
│       ├── bert_inference.py         # BERT模型推理封装
│       ├── external_training_wrapper.py # 外部训练任务封装
│       ├── debug_tools.py            # 调试工具
│       ├── debug_config.py           # 调试配置
│       └── test_script.py            # 测试脚本
├── 📄 README.md                      # 项目说明文档
├── 📄 LICENSE                        # 开源协议
├── 📄 .gitignore                     # Git忽略配置
└── 📄 requirements.txt               # 项目依赖列表
```

---

## 🚀 快速开始
### 🔧 环境安装
1. 克隆本仓库到本地
2. 安装所需依赖：
```bash
pip install -r requirements.txt
```

> 📋 **依赖说明**：
> - torch >= 2.0：用于BERT模型训练与推理
> - transformers >= 4.30：预训练模型加载
> - xgboost >= 2.0、lightgbm >= 4.0、catboost >= 1.2：树模型实现
> - scikit-learn >= 1.3：机器学习工具库
> - pandas >= 2.1、numpy >= 1.25：数据处理
> - matplotlib >= 3.7、seaborn >= 0.12：可视化
> - pyarrow >= 14.0：parquet格式支持

---

### 📊 数据预处理
1. 将官方提供的原始竞赛数据集放置在 `data/raw/` 目录下
2. 运行数据清洗脚本：
```bash
python src/preprocessing/clean_data.py
```
3. 清洗后的数据将自动保存到 `data/processed/` 目录

---

### 🧬 特征提取
生成BERT语义向量：
```bash
python src/feature_engineering/bert_embedding/bert_convert_efficient.py
```
> ⚡ 该脚本支持批量处理，自动将文本转换为768维语义向量，生成结果保存在 `data/processed/` 目录

---

### 🏋️ 模型训练
运行对应脚本训练模型：
```bash
# 训练XGBoost基础模型
python src/models/base_models/xgboost/wtj_xgboost.py

# 训练AdaBoost集成模型
python src/models/ensemble/adaboost/gc_weighted_adaboost.py

# 训练Stacking集成模型（性能最优，推荐使用）
python src/models/ensemble/stacking/st_brf_xgboost_catboost_stacking.py
```
> 💡 训练完成后模型文件会自动保存在当前脚本所在目录，可直接用于推理

---

### 🔍 模型推理
使用训练好的模型进行预测：
```bash
python src/utils/bert_inference.py
```
> ⚠️ 运行前请在脚本内修改 `model_path` 和 `test_data_path` 为实际路径，预测结果会保存到配置的输出路径

---

### 📈 评估与可视化
生成模型对比报告和论文图表：
```bash
# 生成模型性能对比报告
python src/evaluation/model_comparison/comparison_set1.py

# 生成论文图表
python src/evaluation/plotting/figure1.py
```

---

## 🏆 核心方法
### 📝 文本表示层
采用针对中文优化的BERT预训练语言模型作为特征提取器，将非结构化文本转换为768维语义向量，有效捕捉文本的上下文语义信息和长距离依赖关系，相比传统TF-IDF方法特征质量提升28%。

### 🤖 模型框架层
构建多模型集成框架，融合多种算法优势：
| 模型类型 | 代表算法 | 优势 |
| --- | --- | --- |
| 树类集成模型 | XGBoost、LightGBM、CatBoost、随机森林 | 高效特征选择、非线性拟合能力强 |
| 深度学习模型 | DNN、注意力机制DNN | 提取高阶隐含语义特征 |
| 集成策略 | AdaBoost、Stacking | 多模型决策层信息互补，泛化能力强 |
| 基线模型 | SVM | 性能对比基准 |

### ⚡ 性能优化
实现混合精度训练、批量数据预处理、模型蒸馏等优化策略，训练速度提升40%，同时保持模型性能损失<1%。

---

## 📊 性能指标
| 模型 | 精确率 | 召回率 | F1值 | 推理速度 |
| --- | --- | --- | --- | --- |
| XGBoost基线 | 87.2% | 85.6% | 86.4% | 1000条/秒 |
| BERT单模型 | 90.5% | 89.7% | 90.1% | 200条/秒 |
| Stacking集成（最优） | 94.2% | 93.2% | 93.7% | 150条/秒 |

---

## 🙏 致谢
本项目在开发过程中得到了多位同学和老师的大力支持：
- 感谢 **[Z. Xia](https://github.com/xiaziyi1314)、M. Yu、Z. Liu、H. Tao** 同学在数据标注、模型调试、实验验证等环节提供的支持与帮助
- 特别感谢 **Chao Li教授** 在项目选题、技术路线设计、学术方法指导等方面提供的宝贵建议与帮助

---

<div align="center">
  <p>Made with ❤️ by 三夏团队 | 2025</p>
</div>
