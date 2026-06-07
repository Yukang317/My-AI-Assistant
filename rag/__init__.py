# RAG 个人知识库模块
#   rag/
#   ├── config.py           ← 全局配置
#   ├── embedding.py        ← 共享工具
#   ├── parser/             ← 路线1-① 文档解析
#   ├── chunker/            ← 路线1-② 语义分块
#   ├── indexer.py          ← 路线1-③ 摄入编排  🔼 放顶层
#   ├── storage/            ← 外部存储封装
#   ├── retrieval/          ← 路线2-① 检索管线
#   ├── rag_service.py      ← 路线2-② 检索编排
#   └── evaluate.py         ← 评估