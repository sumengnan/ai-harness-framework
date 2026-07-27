from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextBudget:
    """按 token 预算切分上下文各层。

        available      = min(context_window - response_reserve, max_prompt_tokens) - system_tokens
        working_tokens = available × working_ratio        # L1「最近原文」的额度

    两个上限含义不同，别混：
    - context_window / response_reserve —— **物理**约束：塞不下模型会报错/截断。
      response_reserve 按模型的最大输出长度留（思考模型的思维链也算输出）。
    - max_prompt_tokens —— **策略**约束：塞得下，但不划算。典型是分档计价的档位阈值
      （如输入 ≤256K 一个价、超了三倍价）；把阈值填这里即可，不必去谎报 context_window。
      0 = 不设，只受物理窗口约束（默认，与该字段引入前行为一致）。

    ⚠️ working_ratio 只保证「L1 最多用掉 available 的这个比例」，**不保证**剩下那部分真被
    L2/L3 用掉或被约束住：L2 的大小由 context_summary_max_tokens 单独限、L3 由
    context_retrieval_top_k 条数单独限，两者都不看这里的剩余额度。所以把比例调小只是让 L1
    少占，多出来的通常是闲置，并非「留给」谁。
    """
    context_window: int
    response_reserve: int
    working_ratio: float
    max_prompt_tokens: int = 0

    def available(self, system_tokens: int) -> int:
        """可供 L1/L2/L3 分配的额度（已扣掉 system 与给回复预留的空间）。"""
        avail = max(self.context_window - self.response_reserve - system_tokens, 0)
        if self.max_prompt_tokens > 0:
            # max_prompt_tokens 管的是「整个输入」，故同样要扣掉 system 才可比
            avail = min(avail, max(self.max_prompt_tokens - system_tokens, 0))
        return avail

    def working_tokens(self, system_tokens: int) -> int:
        return max(int(self.available(system_tokens) * self.working_ratio), 0)
