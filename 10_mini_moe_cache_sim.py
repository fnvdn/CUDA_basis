import random
from collections import Counter, deque

"""
实验名称：
10_mini_moe_cache_sim

实验目标：
1. 模拟 MoE 中 token 被分配给 expert 的过程。
2. 模拟 GPU 显存中只能缓存一部分 expert。
3. 比较“不预取”和“简单预取”的代价差异。
4. 理解 cache hit、cache miss、LRU、prefetch 的基本含义。

注意：
这个实验不是训练大模型。
它只是一个抽象模拟，用来帮助你理解 MoE expert 预取算法的核心问题。
"""


# expert 总数量
NUM_EXPERTS = 8

# 一共有多少个 batch
NUM_BATCHES = 20

# 每个 batch 有多少个 token
TOKENS_PER_BATCH = 12

# 每个 token 选择几个 expert
TOP_K = 2

# GPU 缓存里最多能放几个 expert
CACHE_SIZE = 4

# 如果 expert 已经在 GPU 缓存里，访问代价较低
COMPUTE_COST = 1.0

# 如果 expert 不在 GPU 缓存里，需要从 CPU/磁盘/远端加载，代价较高
LOAD_COST = 5.0

# 预取 expert 的代价
PREFETCH_COST = 1.0


def make_token_batches():
    """
    构造一批模拟的 MoE 路由结果。

    在真实 MoE 中：
        token 会经过 router，router 决定这个 token 交给哪些 expert 处理。

    在这个实验中：
        我们不用真实 router，而是手动模拟路由结果。

    设计思路：
        前半段 batch 更常使用 expert 0, 1, 2。
        后半段 batch 更常使用 expert 4, 5, 6。

    这样可以模拟真实场景中的“热点 expert 变化”。
    """

    batches = []

    # 初始热点 expert
    hot_experts = [0, 1, 2]

    for step in range(NUM_BATCHES):
        route = []

        for _ in range(TOKENS_PER_BATCH):
            """
            对每个 token，选择 TOP_K 个 expert。

            75% 的概率从热点 expert 里选；
            25% 的概率从所有 expert 里随机选。

            这样可以模拟：
                大部分 token 会集中访问少数热门 expert，
                但也会偶尔访问冷门 expert。
            """
            if random.random() < 0.75:
                chosen = random.sample(hot_experts, TOP_K)
            else:
                chosen = random.sample(range(NUM_EXPERTS), TOP_K)

            route.append(chosen)

        """
        到一半时，热点 expert 发生变化。

        前 10 个 batch：
            热点 expert 是 0, 1, 2

        后 10 个 batch：
            热点 expert 是 4, 5, 6

        这对应真实 MoE 中：
            输入分布变化后，常用 expert 也可能变化。
        """
        if step == NUM_BATCHES // 2:
            hot_experts = [4, 5, 6]

        batches.append(route)

    return batches


class ExpertCache:
    """
    ExpertCache 用来模拟 GPU 显存中的 expert 缓存。

    真实场景：
        expert 参数可能很大，GPU 显存放不下所有 expert。
        所以只能把一部分 expert 放在 GPU 上。

    本实验：
        用一个 set 表示当前缓存中有哪些 expert。
        用 deque 维护 LRU 顺序。
    """

    def __init__(self, capacity):
        """
        capacity:
            缓存最多能放多少个 expert。
        """
        self.capacity = capacity

        # 当前缓存中有哪些 expert
        self.items = set()

        # LRU 队列
        # 左边是最久没用的 expert
        # 右边是最近使用的 expert
        self.lru = deque()

    def has(self, expert_id):
        """
        判断某个 expert 是否已经在缓存中。

        返回：
            True  表示 cache hit
            False 表示 cache miss
        """
        return expert_id in self.items

    def touch(self, expert_id):
        """
        更新 expert 的最近使用状态。

        如果一个 expert 被访问了，
        它就应该变成“最近使用”。
        """
        if expert_id in self.lru:
            self.lru.remove(expert_id)

        self.lru.append(expert_id)

    def load(self, expert_id):
        """
        把一个 expert 加载进缓存。

        如果缓存没满：
            直接放进去。

        如果缓存满了：
            按 LRU 策略淘汰最久没用的 expert。

        返回：
            被淘汰的 expert 编号。
            如果没有淘汰，返回 None。
        """
        evicted = None

        # 如果已经在缓存中，不需要重新加载
        if expert_id in self.items:
            self.touch(expert_id)
            return evicted

        # 如果缓存满了，淘汰最久没用的 expert
        if len(self.items) >= self.capacity:
            evicted = self.lru.popleft()
            self.items.remove(evicted)

        # 加载新 expert
        self.items.add(expert_id)
        self.touch(expert_id)

        return evicted


def prefetch_next_batch(cache, next_route):
    """
    简单预取策略：

    看下一个 batch 的路由结果，
    统计下一个 batch 中哪些 expert 被使用最多。

    然后提前把最常用的几个 expert 放进缓存。

    这不是最强算法，只是一个入门 baseline。

    参数：
        cache:
            当前 expert 缓存。

        next_route:
            下一个 batch 的 token -> expert 路由结果。

    返回：
        预取产生的总代价。
    """

    # 如果已经没有下一个 batch，就不需要预取
    if next_route is None:
        return 0.0

    cost = 0.0

    # 统计下一个 batch 中每个 expert 出现次数
    expert_counter = Counter()

    for expert_ids in next_route:
        expert_counter.update(expert_ids)

    """
    选择出现次数最多的 CACHE_SIZE 个 expert 作为预取候选。

    例如：
        expert 0 出现 10 次
        expert 1 出现 8 次
        expert 2 出现 7 次
        expert 5 出现 2 次

    那就优先预取 0, 1, 2。
    """
    candidates = [
        expert_id
        for expert_id, _ in expert_counter.most_common(CACHE_SIZE)
    ]

    # 把候选 expert 提前加载到缓存中
    for expert_id in candidates:
        if not cache.has(expert_id):
            cache.load(expert_id)
            cost += PREFETCH_COST

    return cost


def run_simulation(batches, use_prefetch):
    """
    运行一次完整模拟。

    参数：
        batches:
            所有 batch 的路由结果。

        use_prefetch:
            False 表示不使用预取。
            True  表示使用简单 next-batch 预取。

    返回：
        一个统计字典，包括：
            总代价
            cache hit 数量
            cache miss 数量
            hit rate
    """

    cache = ExpertCache(CACHE_SIZE)

    total_cost = 0.0
    cache_hits = 0
    cache_misses = 0

    for step, route in enumerate(batches):
        """
        next_route 是下一个 batch 的路由信息。

        如果 use_prefetch=True，
        当前 batch 执行完后，会根据 next_route 提前加载 expert。
        """
        if step + 1 < len(batches):
            next_route = batches[step + 1]
        else:
            next_route = None

        """
        遍历当前 batch 中的每个 token。

        每个 token 会访问 TOP_K 个 expert。
        """
        for token_id, expert_ids in enumerate(route):
            for expert_id in expert_ids:
                if cache.has(expert_id):
                    """
                    cache hit：
                    expert 已经在 GPU 缓存里。
                    不需要加载，只需要计算。
                    """
                    cache_hits += 1
                else:
                    """
                    cache miss：
                    expert 不在 GPU 缓存里。
                    需要先加载 expert，产生 LOAD_COST。
                    """
                    cache_misses += 1
                    cache.load(expert_id)
                    total_cost += LOAD_COST

                """
                访问 expert 后，更新 LRU 状态。
                """
                cache.touch(expert_id)

                """
                模拟 expert 计算代价。

                真实 MoE 中，这里会执行 expert MLP 前向计算。
                本实验中只加一个抽象代价。
                """
                total_cost += COMPUTE_COST

        """
        打印几个关键 step，观察缓存内容变化。
        """
        if step in [0, NUM_BATCHES // 2, NUM_BATCHES - 1]:
            print(
                f"step={step:02d}, "
                f"prefetch={use_prefetch}, "
                f"cache={sorted(cache.items)}"
            )

        """
        当前 batch 执行完后，预取下一个 batch 可能用到的 expert。
        """
        if use_prefetch:
            total_cost += prefetch_next_batch(cache, next_route)

    total_accesses = cache_hits + cache_misses

    if total_accesses > 0:
        hit_rate = cache_hits / total_accesses
    else:
        hit_rate = 0.0

    return {
        "cost": total_cost,
        "hits": cache_hits,
        "misses": cache_misses,
        "hit_rate": hit_rate,
    }


def main():
    """
    主函数。
    """

    # 固定随机种子，保证每次运行结果一致
    random.seed(7)

    print("10_mini_moe_cache_sim")
    print("----------------------------------------")

    print("Config:")
    print(f"NUM_EXPERTS       = {NUM_EXPERTS}")
    print(f"NUM_BATCHES       = {NUM_BATCHES}")
    print(f"TOKENS_PER_BATCH  = {TOKENS_PER_BATCH}")
    print(f"TOP_K             = {TOP_K}")
    print(f"CACHE_SIZE        = {CACHE_SIZE}")
    print("----------------------------------------")

    """
    生成模拟的 token -> expert 路由结果。
    """
    batches = make_token_batches()

    """
    第一组实验：
    不使用预取。
    """
    print("\nNo prefetch")
    no_prefetch = run_simulation(batches, use_prefetch=False)

    """
    第二组实验：
    使用简单 next-batch 预取。
    """
    print("\nWith simple next-batch prefetch")
    with_prefetch = run_simulation(batches, use_prefetch=True)

    """
    打印总结。
    """
    print("\nSummary")
    print("----------------------------------------")
    print(f"no prefetch:   {no_prefetch}")
    print(f"with prefetch: {with_prefetch}")

    saved = no_prefetch["cost"] - with_prefetch["cost"]

    print(f"cost saved:    {saved:.1f}")

    """
    简单解释结果。
    """
    print("\nExplanation")
    print("----------------------------------------")
    print("hits     表示 expert 已经在缓存中，不需要重新加载。")
    print("misses   表示 expert 不在缓存中，需要加载。")
    print("hit_rate 表示缓存命中率。")
    print("cost 越低，说明整体访问和加载代价越小。")
    print("prefetch 的目标是提前加载下一批可能使用的 expert，减少 miss。")


if __name__ == "__main__":
    main()
