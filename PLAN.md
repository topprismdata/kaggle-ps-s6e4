# S6E4 优化流程 — 对抗审查 + 对抗验证

## 流程概览

```
[Research Agent] ──→ 研究报告
       ↓
[对抗审查 Agent] ──→ 质疑研究报告，找出遗漏和风险
       ↓
[Main Agent] ──→ 基于研究报告 + 对抗反馈，制定 R06 方案
       ↓
[对抗审查 Agent] ──→ 审查 R06 代码，检查常见陷阱
       ↓
[Main Agent] ──→ 运行 R06
       ↓
[对抗验证] ──→ 检测 train/test drift，验证 CV-LB 一致性
       ↓
[提交 + 分析]
```

## 两种对抗机制

### 1. 对抗审查 Agent (Red Team Review)

在关键决策点，启动一个独立 agent 进行对抗审查：

**审查时机 A: 方案设计阶段**
- 输入：研究报告 + 主 agent 的 R06 设计方案
- 审查内容：
  - 是否有更简单的方法达到同样效果？
  - 方案中的最大风险是什么？
  - 是否遗漏了 top notebook 的关键技术？
  - 运行时间是否合理？能否缩短？

**审查时机 B: 代码编写后**
- 输入：R06 脚本代码
- 审查内容：
  - ID 列是否正确处理（避免 R05 的 ID bug）
  - TE 是否在 fold 内进行（防泄漏）
  - class_weight 是否正确设置
  - early_stopping 配置是否正确
  - 提交文件格式是否正确

### 2. 对抗验证 (Adversarial Validation)

技术层面的验证：

```python
# 检测 train/test 分布差异
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 构造二分类：train=0, test=1
adv_train = train[features].copy()
adv_test = test[features].copy()
adv_train['is_test'] = 0
adv_test['is_test'] = 1
adv_data = pd.concat([adv_train, adv_test])

X_adv = adv_data[features]
y_adv = adv_data['is_test']

# 如果 AUC > 0.7，说明 train/test 分布差异大
clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X_adv, y_adv, cv=5, scoring='roc_auc')
print(f"Adversarial AUC: {scores.mean():.4f}")
# AUC ≈ 0.5: 无 drift，特征安全
# AUC > 0.7: 有 drift，需要特征选择或调整
```

## R06 优化方案

基于 R05 (LB=0.97765) 的改进：

1. **减少模型数量**：只用 3 XGB + 1 CB = 4 个模型（去掉 LGB）
2. **加入原始数据融合**：以权重 0.35 合并原始数据
3. **加入对抗验证**：检测 train/test drift
4. **优化 TE 速度**：缓存 pairwise 特征，减少重复计算
5. **预期运行时间**：从 7.4 小时降到 ~3 小时
