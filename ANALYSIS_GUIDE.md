# 游戏本地化翻译分析指南

## 文件说明

### 1. `cleaned.json`
- **内容**: 清理后的翻译数据，共5394条记录
- **字段**:
  - `original_en_text`: 英文原文（已移除HTML标签）
  - `original_zh_text`: 官方中文翻译（已移除HTML标签）
  - `translated_text`: 重译版本（已移除HTML标签）
- **数据来源**: Hollow Knight: Silksong 游戏本地化文件

### 2. `clean_json.py`
- **功能**: 从原始数据中提取关键字段，移除HTML标签
- **使用**: `python clean_json.py`（已运行，生成cleaned.json）

### 3. `similarity_calculation_3.py`
- **功能**: 分析官方翻译与重译版本的语义差异
- **特点**:
  - 基于 Qwen3-Embedding 模型
  - 计算语义相似度
  - 评估本土化程度
  - 提供详细的统计分析

## 快速开始

### 运行分析脚本

```bash
python similarity_calculation_3.py
```

### 默认设置
- 分析样本: 100条（随机采样）
- 详细显示: 前10条
- 设备: CPU

### 修改分析规模

编辑 `similarity_calculation_3.py` 第318行：

```python
# 分析100条
triples = load_translation_data('cleaned.json', sample_size=100)

# 分析全部5394条（需要较长时间）
triples = load_translation_data('cleaned.json', sample_size=None)

# 分析500条
triples = load_translation_data('cleaned.json', sample_size=500)
```

## 分析输出

### 1. 单个翻译对比
每个翻译对会显示：
- 英文原文
- 官方中文翻译
- 重译版本
- 语义相似度（带可视化进度条）
- 策略差异度
- 本土化程度

### 2. 统计摘要
- 平均语义相似度
- 平均策略差异
- 平均本土化程度
- 按文本长度分类统计

### 3. 研究结论
- 翻译策略倾向（归化/异化）
- 翻译一致性分析
- 本土化程度对比

## 研究方向

### 1. 专有名词翻译
```python
# 在cleaned.json中查找特定词汇
# 例如: "Pharloom", "Weaver", "Wyrm"
```

### 2. 不同文本类型
- 对话文本
- 诗歌/韵文
- 叙述性文本
- 系统提示

### 3. 翻译策略模式
- 文学性翻译 vs 直译
- 文化适应 vs 源语言保留
- 正式 vs 口语化

## 数据特点

### 游戏类型
- 独立游戏：Hollow Knight: Silksong
- 风格：黑暗奇幻、诗意叙事
- 特点：大量隐喻、专有名词、文学化语言

### 翻译特点
- **官方翻译**: 可能更注重本土化和玩家体验
- **重译版本**: 可能更注重字面准确性

### 适合研究的主题
1. 游戏本地化策略
2. 文学性文本的翻译
3. 专有名词处理
4. 文化适应与异化
5. 翻译一致性

## 扩展分析

### 结合可视化
```bash
# 使用聚类可视化
python 04_text_clustering_visualization.py
```

### 语义搜索
```bash
# 使用语义搜索查找相似翻译
python 03_semantic_search.py
```

## 注意事项

1. **首次运行**: 会自动下载 Qwen3-Embedding 模型（约8-16GB）
2. **运行时间**: 100条样本约需5-10分钟（取决于硬件）
3. **内存需求**: 建议至少8GB可用内存
4. **GPU加速**: 如有GPU，可将 `device='cpu'` 改为 `device='cuda'`

## 数据示例

```json
{
  "original_en_text": "Pilgrim You are blessed to walk...",
  "original_zh_text": "朝圣者 行走于纺络根基是你的殊荣...",
  "translated_text": "朝圣者 您很幸运能够行走在法鲁的基础上..."
}
```

## 问题排查

### 模型加载失败
```bash
# 检查网络连接
# 或使用国内镜像（在config.py中配置）
```

### 内存不足
```bash
# 减少sample_size
# 或使用更小的模型（4B版本）
```

### 编码错误
- 确保所有文件使用UTF-8编码
- Windows用户注意控制台编码设置

## 联系与反馈

如有问题或建议，请参考项目README。


