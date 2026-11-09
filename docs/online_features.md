# 在线集合预测器的固定特征契约

`corem-online-hash-v1` 将请求和所选记忆集合映射到 256 维实数向量，并以
`sigmoid(intercept + weights · features)` 输出成功概率。此文件固定本仓库的完整
特征模板，避免训练与评估采用不同实现。论文仅列出特征类别，没有公布原作者逐项
特征模板；这里的契约不能冒充原作者未公开的参数。

## 哈希和顺序

- 文本以 Unicode `\w+` 分词并转小写，去重后按字典序遍历。
- 每个特征名以 UTF-8 编码，取 SHA-256 前 8 字节的大端整数对 256 取模。
- 哈希碰撞在同一维度相加，不使用随机盐或符号哈希。
- 请求词、原子、版本、记忆 ID、超边 ID 以及交互对均先排序，结果不依赖 Python 哈希种子。

## 特征名和数值

| 类别 | 特征名 | 贡献 |
| --- | --- | --- |
| 请求 | `request:<word>`、`required:<atom>`、`active-version:<version>` | 每个出现项 +1 |
| 单条记忆 | `record-count`、`token-cost` | 选中量 `x_i`；后者为 `x_i × token_cost_i / 160` |
| 记忆内容 | `memory:<word>`、`evidence:<atom>`、`record-version:<version>`、`operation:<id>` | 每个出现项 +`x_i` |
| 请求交叉 | `request-memory:<word>`、`required-evidence:<atom>`、`matching-version:<version>` | 对相交项 +`x_i` |
| 依赖超边 | `dependency:<relation_type>`、`dependency-evidence:<atom>` | 对可用且适用的超边 +`∏_{i∈edge} x_i` |
| 记忆交互 | `cross-record`、`shared-evidence:<atom>`、`duplicate-pair` | 对相应记录对 +`x_i x_j` |

离散检索时，选中记录的 `x_i=1`，其余记录不存在。留存松弛计算时，`x_i∈[0,1]`，
模型对各 `x_i` 使用同一多项式特征的解析导数。硬冲突在检索约束中检查，不作为成功
概率的软特征。训练只接受实际执行且经独立检查的开发记录，使用 BCE 梯度和 L2
正则；评估期间权重固定。训练参数与模型文件分开记录在 provenance 文件中。
