import math
import random
from collections import Counter, deque

"""
实验名称：
11_mini_moe_mlp

实验目标：
1. 在 10_mini_moe_cache_sim 的基础上，把“假 expert”换成小型 MLP expert。
2. 模拟 token -> router -> top-k expert -> expert MLP -> output 的 MoE 推理流程。
3. 继续保留 expert cache、cache hit、cache miss、prefetch 的概念。
4. 不依赖 PyTorch，使用纯 Python 列表实现，方便先跑通逻辑。

重要说明：
这个实验仍然不是大模型。
它是一个最小 MoE 推理结构，用来理解最终研究中的几个核心部件：
    token
    router
    expert
    MLP
    cache
    prefetch
"""


# expert 总数量
NUM_EXPERTS = 8

# 输入向量维度
HIDDEN_SIZE = 8

# expert MLP 中间层维度
FFN_SIZE = 16

# batch 数量
NUM_BATCHES = 20

# 每个 batch 的 token 数量
TOKENS_PER_BATCH = 12

# 每个 token 选择几个 expert
TOP_K = 2

# GPU expert cache 最多放几个 expert
CACHE_SIZE = 4

# 缓存命中后执行 expert 的抽象计算代价
COMPUTE_COST = 1.0

# expert 不在缓存中，需要加载 expert 的抽象代价
LOAD_COST = 5.0

# 预取 expert 的抽象代价
PREFETCH_COST = 1.0


def relu(x):
    """
    ReLU 激活函数。

    如果 x > 0，返回 x。
    如果 x <= 0，返回 0。

    在真实 MLP 中，线性层之间通常会加非线性激活函数。
    """
    return max(0.0, x)


def dot(a, b):
    """
    向量点积。

    例如：
        a = [1, 2, 3]
        b = [4, 5, 6]

        dot(a, b) = 1*4 + 2*5 + 3*6
    """
    total = 0.0
    for x, y in zip(a, b):
        total += x * y
    return total


def matvec(matrix, vector):
    """
    矩阵乘向量。

    matrix:
        二维列表，形状可以理解成 out_dim x in_dim。

    vector:
        一维列表，长度是 in_dim。

    返回：
        一维列表，长度是 out_dim。

    这相当于神经网络里的一个线性层：
        y = W x
    """
    return [dot(row, vector) for row in matrix]


class TinyMLPExpert:
    """
    一个小型 MLP expert。

    结构：
        input
          ↓
        Linear 1
          ↓
        ReLU
          ↓
        Linear 2
          ↓
        output

    真实 MoE 中，每个 expert 通常也是一个 FFN/MLP。
    这里用很小的矩阵模拟它。
    """

    def __init__(self, expert_id):
        self.expert_id = expert_id

        """
        为了让每个 expert 有不同参数，
        我们用 expert_id 影响随机种子。
        """
        rng = random.Random(1000 + expert_id)

        """
        W1:
            第一层权重。
            形状：FFN_SIZE x HIDDEN_SIZE。

        W2:
            第二层权重。
            形状：HIDDEN_SIZE x FFN_SIZE。
        """
        self.w1 = [
            [rng.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)]
            for _ in range(FFN_SIZE)
        ]

        self.w2 = [
            [rng.uniform(-0.5, 0.5) for _ in range(FFN_SIZE)]
            for _ in range(HIDDEN_SIZE)
        ]

    def forward(self, token):
        """
        执行 expert 的前向计算。

        token:
            一个长度为 HIDDEN_SIZE 的输入向量。

        返回：
            一个长度为 HIDDEN_SIZE 的输出向量。
        """
        hidden = matvec(self.w1, token)
        hidden = [relu(x) for x in hidden]
        output = matvec(self.w2, hidden)
        return output


def make_experts():
    """
    创建所有 expert。

    在真实大模型中：
        expert 参数可能非常大，不一定全部放在 GPU 上。

    在这个实验中：
        expert 很小，所以我们全部放在 CPU 内存里。
        cache 只是在逻辑上模拟“哪些 expert 当前在 GPU 上”。
    """
    return [TinyMLPExpert(expert_id) for expert_id in range(NUM_EXPERTS)]


def make_token():
    """
    创建一个 token 向量。

    真实模型中 token 通常是 embedding 向量。
    这里用随机向量模拟。
    """
    return [random.uniform(-1.0, 1.0) for _ in range(HIDDEN_SIZE)]


def make_batches():
    """
    构造 token batch 和路由结果。

    每个 batch 包含：
        tokens:
            一组 token 向量。

        route:
            每个 token 对应的 top-k expert 编号。

    前半段更常访问 expert 0, 1, 2。
    后半段更常访问 expert 4, 5, 6。
    """
    batches = []
    hot_experts = [0, 1, 2]

    for step in range(NUM_BATCHES):
        tokens = [make_token() for _ in range(TOKENS_PER_BATCH)]
        route = []

        for _ in range(TOKENS_PER_BATCH):
            if random.random() < 0.75:
                chosen = random.sample(hot_experts, TOP_K)
            else:
                chosen = random.sample(range(NUM_EXPERTS), TOP_K)
            route.append(chosen)

        if step == NUM_BATCHES // 2:
            hot_experts = [4, 5, 6]

        batches.append((tokens, route))

    return batches


class ExpertCache:
    """
    expert cache。

    用来模拟 GPU 显存中当前放着哪些 expert。
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.items = set()
        self.lru = deque()

    def has(self, expert_id):
        return expert_id in self.items

    def touch(self, expert_id):
        """
        将 expert 标记为最近使用。
        """
        if expert_id in self.lru:
            self.lru.remove(expert_id)
        self.lru.append(expert_id)

    def load(self, expert_id):
        """
        加载 expert。

        如果 cache 满了，就淘汰最久没使用的 expert。
        """
        evicted = None

        if expert_id in self.items:
            self.touch(expert_id)
            return evicted

        if len(self.items) >= self.capacity:
            evicted = self.lru.popleft()
            self.items.remove(evicted)

        self.items.add(expert_id)
        self.touch(expert_id)
        return evicted


def prefetch_next_batch(cache, next_route):
    """
    根据下一个 batch 的路由结果做简单预取。

    策略：
        统计下一个 batch 中出现次数最多的 expert。
        提前把它们加载进 cache。
    """
    if next_route is None:
        return 0.0

    cost = 0.0
    counter = Counter()

    for expert_ids in next_route:
        counter.update(expert_ids)

    candidates = [
        expert_id
        for expert_id, _ in counter.most_common(CACHE_SIZE)
    ]

    for expert_id in candidates:
        if not cache.has(expert_id):
            cache.load(expert_id)
            cost += PREFETCH_COST

    return cost


def average_vectors(vectors):
    """
    对多个 expert 输出求平均。

    因为每个 token 会经过 TOP_K 个 expert，
    所以需要把多个 expert 的输出合并成一个输出。

    真实 MoE 中通常还会乘 router gate 权重。
    这里为了简单，直接求平均。
    """
    output = [0.0 for _ in range(HIDDEN_SIZE)]

    for vec in vectors:
        for i in range(HIDDEN_SIZE):
            output[i] += vec[i]

    for i in range(HIDDEN_SIZE):
        output[i] /= len(vectors)

    return output


def vector_norm(vector):
    """
    计算向量模长。

    用来打印一个简单的输出摘要，确认 MLP 真的做了计算。
    """
    return math.sqrt(sum(x * x for x in vector))


def run_simulation(batches, experts, use_prefetch):
    """
    运行一次 MoE MLP 推理模拟。
    """
    cache = ExpertCache(CACHE_SIZE)

    total_cost = 0.0
    cache_hits = 0
    cache_misses = 0
    last_output_norm = 0.0

    for step, (tokens, route) in enumerate(batches):
        if step + 1 < len(batches):
            next_route = batches[step + 1][1]
        else:
            next_route = None

        batch_outputs = []

        for token_id, expert_ids in enumerate(route):
            token = tokens[token_id]
            expert_outputs = []

            for expert_id in expert_ids:
                if cache.has(expert_id):
                    cache_hits += 1
                else:
                    cache_misses += 1
                    cache.load(expert_id)
                    total_cost += LOAD_COST

                cache.touch(expert_id)

                """
                这里和第 10 个实验不同：
                第 10 个实验只加了一个抽象 COMPUTE_COST。
                这里真的调用 expert.forward(token)，执行一个小型 MLP。
                """
                expert_output = experts[expert_id].forward(token)
                expert_outputs.append(expert_output)

                total_cost += COMPUTE_COST

            token_output = average_vectors(expert_outputs)
            batch_outputs.append(token_output)

        """
        打印几个关键 step 的 cache 状态和输出摘要。
        """
        if step in [0, NUM_BATCHES // 2, NUM_BATCHES - 1]:
            last_output_norm = vector_norm(batch_outputs[0])
            print(
                f"step={step:02d}, "
                f"prefetch={use_prefetch}, "
                f"cache={sorted(cache.items)}, "
                f"first_output_norm={last_output_norm:.4f}"
            )

        if use_prefetch:
            total_cost += prefetch_next_batch(cache, next_route)

    total_accesses = cache_hits + cache_misses
    hit_rate = cache_hits / total_accesses if total_accesses else 0.0

    return {
        "cost": total_cost,
        "hits": cache_hits,
        "misses": cache_misses,
        "hit_rate": hit_rate,
        "last_output_norm": last_output_norm,
    }


def main():
    random.seed(7)

    print("11_mini_moe_mlp")
    print("----------------------------------------")
    print("This is a tiny MoE inference simulation with real MLP experts.")
    print("No PyTorch is required in this version.")
    print("----------------------------------------")

    print("Config:")
    print(f"NUM_EXPERTS       = {NUM_EXPERTS}")
    print(f"HIDDEN_SIZE       = {HIDDEN_SIZE}")
    print(f"FFN_SIZE          = {FFN_SIZE}")
    print(f"NUM_BATCHES       = {NUM_BATCHES}")
    print(f"TOKENS_PER_BATCH  = {TOKENS_PER_BATCH}")
    print(f"TOP_K             = {TOP_K}")
    print(f"CACHE_SIZE        = {CACHE_SIZE}")
    print("----------------------------------------")

    experts = make_experts()
    batches = make_batches()

    print("\nNo prefetch")
    no_prefetch = run_simulation(batches, experts, use_prefetch=False)

    print("\nWith simple next-batch prefetch")
    with_prefetch = run_simulation(batches, experts, use_prefetch=True)

    print("\nSummary")
    print("----------------------------------------")
    print(f"no prefetch:   {no_prefetch}")
    print(f"with prefetch: {with_prefetch}")
    print(f"cost saved:    {no_prefetch['cost'] - with_prefetch['cost']:.1f}")

    print("\nExplanation")
    print("----------------------------------------")
    print("expert 现在不只是编号，而是一个两层 MLP。")
    print("router 的作用由 route 列表模拟：每个 token 选择 TOP_K 个 expert。")
    print("cache hit 表示 expert 已经在模拟 GPU cache 中。")
    print("cache miss 表示 expert 需要加载。")
    print("prefetch 的目标是在 expert 真正被用到之前提前加载它。")


if __name__ == "__main__":
    main()
