from __future__ import annotations

import logging

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class HarnessConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HARNESS_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    # 可选数值项：允许在 .env 里写成空串表示「不设」（= None），避免空值被当成非法整数/浮点。
    @field_validator("max_tokens_budget", "max_wall_seconds", mode="before")
    @classmethod
    def _blank_to_none(cls, v):
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    # 温度入口统一夹到 [0,1]（见 llm/sampling.py 为何上限不是协议的 2）。写超了就地纠正
    # 并告警，而不是留到发请求时被端点拒——那时报错信息只会是一句「invalid parameter」。
    @field_validator("temperature")
    @classmethod
    def _clamp_temperature(cls, v):
        from .llm.sampling import TEMP_MAX, TEMP_MIN, clamp_temperature
        c = clamp_temperature(v)
        if c != float(v):
            logging.getLogger(__name__).warning(
                "HARNESS_TEMPERATURE=%s 超出 [%s, %s]，已夹到 %s", v, TEMP_MIN, TEMP_MAX, c)
        return c

    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    system_prompt: str = "You are a helpful assistant."
    max_steps: int = 100
    # 循环/停滞检测：连续多少步发起完全相同的工具调用（同名+同参）判为原地打转——先注入一次
    # 纠偏提示让模型换思路，纠偏后仍重复才中止；<2 关闭。防模型卡在重复动作上白跑到 max_steps。
    loop_detect_window: int = 3
    # 采样基准温度。未被角色档/意图路由覆盖时用它（如聊天闲聊轮）。恒被夹在 [0,1]：
    # 上限取 1 而非协议的 2，因为 Anthropic 只到 1、百炼不收 2，且 >1 只会让输出退化。
    temperature: float = 0.7
    # 角色温度覆盖 {角色名: 温度}，合并进 app/sampling_policy.py 的内置默认表（该表列了
    # 全部 18 处机械/判断类调用点的建议值）。例：{"judge":0.0,"quiz_generate":0.9}
    role_temperatures: dict = {}
    # 意图温度覆盖 {意图名: 温度}，同上，作用于面向用户生成的那几处（主循环/直答/汇总）。
    intent_temperatures: dict = {}
    # 运行期动态增减温度（打转纠偏升温、重答升温、结构解析失败降温）。关掉则只保留静态分档。
    enable_dynamic_temperature: bool = True
    # 不接受 temperature/top_p 的模型（子串匹配模型名）：命中则一个采样参数都不发。
    # 部分推理模型（o1 系、思考模式下的一些 Qwen）会因「不支持的参数」直接报错。
    sampling_unsupported_models: list = []
    request_timeout: float = 60.0
    # 透传给 chat.completions.create 的额外请求体（默认空=不改变行为）。用于开关厂商私有参数，
    # 例如 Qwen3 关闭「思考模式」提速：HARNESS_LLM_EXTRA_BODY={"enable_thinking": false}
    # （DashScope 百炼 compatible-mode 用此形式；自托管 vLLM 用
    # {"chat_template_kwargs": {"enable_thinking": false}}）。
    llm_extra_body: dict = {}
    # 不支持「思考模式」切换参数的模型（子串匹配模型名，JSON 数组）：命中的模型不发
    # enable_thinking / thinking，避免端点因「未知参数」报错。例：["qwen-turbo","-flash"]
    thinking_unsupported_models: list = []
    max_retries: int = 2
    retry_base_delay: float = 0.5
    max_tokens_budget: int | None = None
    max_wall_seconds: float | None = None
    # 单个工具结果回喂给模型的全局字符上限（所有工具的二次截断兜底）。放到 1M 以让
    # read_file 等大输出真正透传（各工具仍受自身上限约束，如 shell 输出走 sandbox_output_max_chars）。
    tool_result_max_chars: int = 1_000_000
    include_usage: bool = True
    otel_enabled: bool = False
    otel_exporter: str = "console"      # console | otlp
    otel_endpoint: str = ""
    price_map: dict = {}                 # {model: [in_per_1k, out_per_1k]}（扁平计费，按模型 key，实时口径）
    # 当前模型分层计费：按「输入长度」分档，每档 [输入上限tokens, 输入价/百万token, 输出价/百万token]，
    # 升序排列，末档为封顶价；空=未知（成本显示 —）。默认 qwen-plus 档位（¥/百万 token）：
    # 输入≤256K=1.6、256K~1M=4.8；输出≤256K=6.4、256K~1M=19.2。
    model_price_tiers: list = [[256000, 1.6, 6.4], [1000000, 4.8, 19.2]]
    # 按模型的分层计费覆盖：{模型名: tiers}。给主/快速/embedding/rerank/judge 各配一份，未命中的
    # 模型回退上面的 model_price_tiers（默认档）。例：
    #   {"qwen-plus":[[256000,1.6,6.4],[1000000,4.8,19.2]], "qwen-turbo":[[1000000,0.3,0.6]]}
    model_price_tiers_by_model: dict = {}
    price_currency: str = "¥"            # 估算成本显示的货币符号
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""          # 空则回退用 api_key
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    memory_db_path: str = "memory.db"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_hard_max: int = 2000        # 切分容量硬上限：表格/代码块在此上限内整块保留，绝不超此值
    search_top_k: int = 10
    memory_collection: str = "knowledge"
    retrieval_candidate_pool: int = 20
    retrieval_w_relevance: float = 1.0
    retrieval_w_recency: float = 0.2
    retrieval_w_importance: float = 0.1
    retrieval_recency_half_life_days: float = 30.0
    retrieval_use_keyword: bool = True
    retrieval_use_mmr: bool = True
    retrieval_mmr_lambda: float = 0.7
    retrieval_rrf_k: int = 60
    # 查询期召回增强（默认全关=零行为变更）；开启需 embedding 已配（有 memory）。
    # 三路各加一路召回并入 RRF，共享一次查询期 LLM 规划（QueryPlanner），带超时降级。
    retrieval_use_entity_recall: bool = False   # LLM 从 query 抽实体键 → list_by_entity 精确取
    retrieval_use_multi_query: bool = False      # LLM 改写多条等价查询各跑向量召回
    retrieval_use_hyde: bool = False             # LLM 生成假设答案文档 → 其向量召回
    retrieval_multi_query_n: int = 3
    retrieval_query_plan_timeout_s: float = 2.0  # 规划 LLM 超时（首字关键路径，超时即降级）
    # 精排（rerank）：默认关=维持现状（NoOpReranker）。开启且配了端点+模型才生效，
    # 全局作用于所有检索路径（知识库/题库/对话记忆/长期记忆）。端点须为
    # OpenAI/Cohere/Jina 兼容的 POST {base}/rerank。失败自动降级为原序，不打断检索。
    enable_rerank: bool = False
    rerank_style: str = "openai"         # openai（Cohere/Jina/SiliconFlow 兼容）| dashscope（千问 qwen）
    rerank_base_url: str = ""             # openai 风格填到 /v1；dashscope 填完整 endpoint
    rerank_api_key: str = ""             # 空则回退 embedding_api_key → api_key
    rerank_model: str = ""               # 如 BAAI/bge-reranker-v2-m3 或 qwen3-rerank
    rerank_timeout: float = 30.0
    rerank_top_n: int = 0                # 0=送全部候选精排；>0 只精排前 N（省调用成本）
    # 相关性下限：精排分低于它的候选直接丢弃（0=关闭）。这是整条检索链上唯一的绝对相关性
    # 信号——RRF 只看排名、加权前又做了候选集内 min-max 归一化（最好的那条永远得 1.0），
    # 所以不靠它就没有任何一处能表达「都不够相关」，小知识库会被任意 query 整个倒出来。
    # 量纲随精排模型而变（[0,1] 概率 vs 未归一化 logit），故默认关闭，按实测配置。
    # qwen3-rerank 实测：无关 query 最高 0.26，相关 query 最低 0.43 → 0.35 落在间隔中央。
    rerank_min_score: float = 0.0
    # 容器沙箱
    sandbox_backend: str = "local"          # local | docker
    sandbox_docker_host: str = ""           # tcp://host:2376（Docker daemon 的 TLS 端口）
    # 直连 Docker daemon TLS 端口的双向 TLS 证书（不再走 SSH）
    sandbox_docker_tls_ca_cert: str = ""        # CA 证书路径（校验服务端）
    sandbox_docker_tls_client_cert: str = ""    # 客户端证书路径
    sandbox_docker_tls_client_key: str = ""     # 客户端私钥路径
    sandbox_docker_tls_verify: bool = True      # 是否校验服务端证书
    # run_shell 与「缺省（未指定 language）的 write_file/read_file/list_files」落这个 shell 容器；
    # 沙箱内的 http_request（curl）与 DNS 解析（getent）也在此容器执行——故 shell 镜像须同时自带
    # curl 与 glibc 的 getent。CentOS/RHEL 系基础镜像两者都有；debian/ubuntu slim 不含 curl、
    # alpine 不含 getent，都不合适。默认用 centos:stream9（重构前的基础镜像，已验证可用）。
    sandbox_shell_image: str = "quay.io/centos/centos:stream9"
    # 语言[+版本]->镜像：run_python/run_node/run_java 按语言[+可选 version]自动起对应语言容器执行，
    # 各语言容器各自独立工作区、按 (会话,语言) 缓存复用、1h 空闲销毁。key 优先 f"{language}{version}"
    # （回退 language）；显式指定的 version 无对应镜像则报错。write_file 等带 language 参数时也落对应容器。
    # 含 java 多版本，供 run_java 的 version 选择。
    sandbox_lang_images: dict = {"python":"python:3.12-slim","node":"node:20-slim","java":"eclipse-temurin:21-jdk",
          "java8":"eclipse-temurin:8-jdk","java11":"eclipse-temurin:11-jdk",
          "java17":"eclipse-temurin:17-jdk","java21":"eclipse-temurin:21-jdk"}
    sandbox_approval_timeout: float = 120.0      # 危险命令人工确认超时（秒）；超时自动拒绝
    sandbox_workspace: str = "/workspace"
    sandbox_user: str = "1000:1000"
    sandbox_network: str = "bridge"              # 所有语言容器统一的网络（bridge=联网，下载落工作区；none=禁网）
    sandbox_mem_limit: str = "200m"              # 每个容器内存上限
    sandbox_cpus: float = 1.0                    # 每个容器 CPU 上限（核）
    # 工作区磁盘上限（/workspace tmpfs 大小）。注意：tmpfs 是内存盘，其占用**算进 mem_limit**——
    # 故实际可写 ≈ min(disk_limit, mem_limit - 进程开销)。默认 500m 是天花板，mem_limit 才是更紧的实际卡口。
    sandbox_disk_limit: str = "500m"
    sandbox_pids_limit: int = 128                # 每个容器进程数上限
    sandbox_read_only: bool = False         # 容器根文件系统是否只读（默认可写）
    # 单次 run_python/run_node/run_java/run_shell 的执行超时（秒）。容器内用 `timeout` 命令强制，
    # 到点即杀（exit 124）——**也罩住你在代码里起的 pip/子进程**，故 pip install 也吃这个预算。
    # 默认 600（10 分钟，给 pip --target 装依赖留足时间）；调小可让失控代码更早被杀。
    sandbox_exec_timeout: float = 600.0
    # 沙箱输出上限（字符）：read_file 读文件、run_shell/run_python 等执行输出共用此上限，
    # 超出截断。放到 1M 以支持读大文件（之前 8000 太小）。注意还受全局 tool_result_max_chars
    # 二次截断，故那个也需 ≥ 此值才真正生效。
    sandbox_output_max_chars: int = 1_000_000
    # 语言容器空闲驱逐（秒）：每个 (会话,语言) 容器按此空闲超时缓存复用——超过此时长无操作才销毁，
    # 有操作即续期；避免每次执行都重建镜像容器。<=0 关闭空闲驱逐（用完即销毁）。默认 1 小时。
    # 与「删除会话即销毁该会话全部语言容器」的主路径无关（那是确定性回收）。
    sandbox_idle_timeout: float = 3600.0
    # 浏览器沙箱空闲驱逐（秒）：浏览器子沙箱是**全局共用一个**（跨会话），懒加载启动、复用，
    # 超过此时长无抓取才销毁（下次用再重建）；进程关停时一并关闭。默认 24 小时。<=0 关闭空闲驱逐。
    browser_sandbox_idle_timeout: float = 86400.0
    # 外部 API/HTTP
    http_allowed_domains: list = []         # 空=放行公网；非空=仅白名单
    http_block_private: bool = True         # SSRF：拦截内网/元数据
    http_timeout: float = 30.0
    http_max_response_bytes: int = 10_000_000
    http_max_redirects: int = 5
    # 空 UA 是最典型的爬虫特征之一，不少站点据此直接 403——此前本工具一个 UA 都不发。
    # 默认按「行为良好的爬虫」惯例如实标明身份（Googlebot 也是这个格式），能解决「仅因为
    # 没有 UA 而被拒」的那批站点。设为空字符串则退回不发 UA。
    # 注：这不会绕过 Cloudflare 之类的 bot 检测——那条路本就走浏览器兜底。若确需伪装成
    # 浏览器，自行用 HARNESS_HTTP_USER_AGENT 覆盖（是否合规由使用者自行判断）。
    http_user_agent: str = "Mozilla/5.0 (compatible; AI-Learning-Helper/1.0; +harness)"
    # 浏览器
    browser_headless: bool = True
    browser_nav_timeout: float = 30.0
    browser_wait_until: str = "networkidle"   # load | domcontentloaded | networkidle
    browser_output_max_chars: int = 8000
    browser_user_agent: str = ""
    # 浏览器专用沙箱镜像（须含 Playwright+Chromium+curl，如 mcr playwright 镜像 + curl）。
    # 配了则每次抓取在该镜像的一次性子沙箱内跑 Chromium，基础镜像可保持轻量（如 python:3.12
    # 无需装 playwright）；留空则复用基础容器（需基础镜像自带 playwright，否则 browse 会报
    # ModuleNotFoundError: No module named 'playwright'）。
    browser_sandbox_image: str = "ai-learning-helper/playwright-py:v1.47.0"
    # 浏览器子沙箱的内存上限：Chromium 远比一般沙箱吃内存，若沿用基础沙箱的小额度
    # （如 100m）会被 OOM 杀掉，容器中途消失、browse 失败。故单列一档，默认 1g。
    browser_sandbox_mem_limit: str = "1g"
    # 在沙箱容器内跑无头 Chromium 时的启动参数（cap_drop=ALL/非 root/小 shm 下必备）
    sandbox_browser_launch_args: list = [
        "--no-sandbox", "--disable-setuid-sandbox",
        "--disable-dev-shm-usage", "--disable-gpu",
    ]
    # 技能（渐进式披露）
    skills_dir: str = "skills"            # 技能目录：<skills_dir>/<name>/SKILL.md
    skill_resource_max_chars: int = 8000  # read_skill_resource 单次读取上限
    # 多 Agent 编排
    agents_dir: str = "agents"        # 子 agent 花名册目录：<agents_dir>/<name>.yaml
    max_dispatch_depth: int = 2       # agent 树最大层数（防无限递归）
    sub_agent_max_steps: int = 100    # 子 agent 单次 run 步数上限
    # 情景记忆
    episode_collection: str = "episodes"
    episode_recall_k: int = 3
    # 持久化
    persistence_db_path: str = "harness.db"
    # MCP（Model Context Protocol）客户端：连接外部/内置 MCP server，把远程工具暴露为本地工具。
    # server 清单在下面这个 JSON 文件里声明（stdio + streamable-http 双传输）；改配置后重启
    # （或调 POST /api/mcp/reload）生效。总开关 enable_mcp 在 AppConfig。
    mcp_config_path: str = "mcp/mcp_servers.json"   # server 清单文件路径
    mcp_connect_timeout: float = 15.0           # 单 server 连接/初始化超时（秒），超时跳过
