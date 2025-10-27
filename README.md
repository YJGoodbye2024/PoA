principle_situation.py 文件里有列表

## google_search.py

--limit <N>
限制只搜索前 N 条 principle，用于测试

--sequential
限制是否异步 request

--output-dir
输出文件夹路径，默认为 Dataset/gemini_references

--principles
后面跟要处理的原则名（可多条），名称之间用空格分隔即可。用于测试

## build_semantic_scholar_archive.py

实现从 responses_parsed.json 到结构化索引与 PDF 的完整流水线：

- 逐条解析引用，提取 DOI/标题提示，构建待查询任务。
- 访问 Semantic Scholar Graph API（自动重试、速率限制间隔 0.2s），复用已有元数据并按 paperId/DOI/标题生成去重 paper_key。
- 将新元数据写入
  Dataset/semantic_scholar/papers/<paper_key>.json，可选下载开放获取 PDF 到 Dataset/semantic_scholar/pdfs/<paper_key>.pdf，失败会记录日志。
- 生成/更新
  Dataset/semantic_scholar/principles.json，每个原则对应 50 个 {paper_key, status} 索引。
- 将未命中、错误信息追加到 Dataset/semantic_scholar/logs/not_found.jsonl 与 errors.jsonl。

运行前请设定 S2_API_KEY，可用 --principles、--limit、--skip-pdf 控制范围与是否抓取 PDF。

--principles <名称 1> <名称 2> ...
只处理指定的原则，名称要与 responses_parsed.json 里的键完全一致。
例：python build_semantic_scholar_archive.py --principles "actor observer asymmetry" "defensive attribution hypothesis"

--limit <N>
限制每个原则最多处理前 N 条引用（用于小样本调试）。
例：python build_semantic_scholar_archive.py --limit 10

--skip-pdf
添加该开关后，即使抓到了开放获取链接也不会下载 PDF，只写元数据。
例：python build_semantic_scholar_archive.py --skip-pdf

--pdf-workers
启动 N 个异步下载 worker。队列里只要有待下载的 PDF，就会被最多 N 个 worker 并行处理，所以下载任务最多同时进行 N 条。

--deep-workers
Number of concurrent deep search workers (default: 10)
启动 N 个深度检索 worker。每个 worker 独立消费 deep search 队列；因此当 --deep-workers=10 时，最多会有 10 个 deep search 请求在同一时间执行。
