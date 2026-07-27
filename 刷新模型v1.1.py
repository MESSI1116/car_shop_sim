import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="每日车辆商店销量模拟器", layout="wide")

st.title("每日车辆商店销量模拟器")

# =========================
# 基础数据
# =========================

player_lookup = {
    1: 301732,
    2: 349292,
    3: 375871,
    4: 396054,
    5: 412749,
    6: 427006,
    7: 439680,
    8: 451328,
    9: 462181,
    10: 472501,
}

tiers = ["T1", "T2", "T3", "T4", "T5"]
segments = ["小R", "中R", "大R", "超R"]

segment_active_counts = {
    "小R": 582662,
    "中R": 17500,
    "大R": 16721,
    "超R": 4427,
}

segment_total = sum(segment_active_counts.values())
segment_weights = {
    seg: segment_active_counts[seg] / segment_total
    for seg in segments
}

segment_avg_total_pay = {
    "小R": 452.72,
    "中R": 6976.33,
    "大R": 20725.47,
    "超R": 116757.12,
}

default_car_count = {
    "T1": 5,
    "T2": 15,
    "T3": 35,
    "T4": 30,
    "T5": 15,
}

default_refresh_weight = {
    "T1": 5.0,
    "T2": 15.0,
    "T3": 35.0,
    "T4": 30.0,
    "T5": 15.0,
}

default_attractiveness = {
    "T1": 2.0,
    "T2": 1.5,
    "T3": 1.0,
    "T4": 0.6,
    "T5": 0.3,
}

default_tier_waiting_penalty_strength = {
    "T1": 0.20,
    "T2": 0.40,
    "T3": 0.70,
    "T4": 1.00,
    "T5": 1.20,
}

default_segment_value_coef = {
    "小R": 1.0,
    "中R": 1.4,
    "大R": 2.0,
    "超R": 3.0,
}

default_segment_price_sigma_coef = {
    "小R": 0.15,
    "中R": 0.25,
    "大R": 0.40,
    "超R": 0.60,
}

default_segment_fund_sigma_coef = {
    "小R": 0.20,
    "中R": 0.35,
    "大R": 0.60,
    "超R": 1.00,
}

default_segment_hesitation_coef = {
    "小R": 0.20,
    "中R": 0.10,
    "大R": 0.05,
    "超R": 0.00,
}


# =========================
# 工具函数
# =========================

def allocate_counts(total_count, weights_dict, ordered_keys):
    raw = np.array([total_count * weights_dict[k] for k in ordered_keys])
    base = np.floor(raw).astype(int)
    remainder = int(total_count - base.sum())

    if remainder > 0:
        fractions = raw - base
        order = np.argsort(-fractions)
        for idx in order[:remainder]:
            base[idx] += 1

    return {
        k: int(v)
        for k, v in zip(ordered_keys, base)
    }


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -60, 60)))


# =========================
# 侧边栏参数
# =========================

st.sidebar.header("基础模拟参数")

cycle_days = st.sidebar.slider(
    "刷新周期：每几天刷新一次",
    min_value=1,
    max_value=10,
    value=7
)

players = player_lookup[cycle_days]

st.sidebar.metric(
    "该周期内去重登录小R及以上玩家数",
    f"{players:,}"
)

simulate_cycles = st.sidebar.number_input(
    "模拟周期数",
    min_value=1,
    max_value=100,
    value=10
)
simulate_cycles = int(simulate_cycles)

total_days = simulate_cycles * cycle_days

st.sidebar.metric(
    "总模拟天数",
    f"{total_days:,} 天"
)

cars_per_cycle = st.sidebar.slider(
    "每个玩家每周期刷到车辆数",
    min_value=1,
    max_value=20,
    value=3
)

price = st.sidebar.number_input(
    "单车价格",
    min_value=0.0,
    value=68.0
)

random_seed = st.sidebar.number_input(
    "随机种子",
    min_value=0,
    max_value=999999,
    value=42
)
random_seed = int(random_seed)

st.sidebar.header("玩家资金模型参数")

fixed_period_days = st.sidebar.number_input(
    "固定资金周期天数",
    min_value=1,
    max_value=365,
    value=30
)
fixed_period_days = int(fixed_period_days)

small_r_budget_per_period = st.sidebar.number_input(
    "小R每固定周期车辆预算",
    min_value=0.0,
    value=68.0
)

ability_alpha = st.sidebar.slider(
    "付费能力压缩指数 α",
    min_value=0.0,
    max_value=1.0,
    value=0.40,
    step=0.01
)

initial_fund_ratio = st.sidebar.slider(
    "初始资金比例",
    min_value=0.0,
    max_value=2.0,
    value=0.50,
    step=0.05
)

allow_negative_funds = st.sidebar.checkbox(
    "允许资金为负",
    value=True
)

st.sidebar.header("双Sigmoid参数")

base_psychological_value = st.sidebar.number_input(
    "基准心理价位",
    min_value=0.0,
    value=68.0
)

segment_value_coef = {}
segment_price_sigma_coef = {}
segment_fund_sigma_coef = {}
segment_hesitation_coef = {}

for seg in segments:
    with st.sidebar.expander(f"{seg} 参数", expanded=False):
        segment_value_coef[seg] = st.number_input(
            f"{seg} 心理价值系数",
            min_value=0.0,
            max_value=20.0,
            value=default_segment_value_coef[seg],
            step=0.05,
            key=f"{seg}_value_coef"
        )

        segment_price_sigma_coef[seg] = st.number_input(
            f"{seg} 价格σ系数",
            min_value=0.01,
            max_value=10.0,
            value=default_segment_price_sigma_coef[seg],
            step=0.01,
            key=f"{seg}_price_sigma_coef"
        )

        segment_fund_sigma_coef[seg] = st.number_input(
            f"{seg} 资金σ系数",
            min_value=0.01,
            max_value=10.0,
            value=default_segment_fund_sigma_coef[seg],
            step=0.01,
            key=f"{seg}_fund_sigma_coef"
        )

        segment_hesitation_coef[seg] = st.number_input(
            f"{seg} 犹豫成本系数",
            min_value=0.0,
            max_value=5.0,
            value=default_segment_hesitation_coef[seg],
            step=0.01,
            key=f"{seg}_hesitation_coef"
        )

st.sidebar.header("刷新心理参数")

future_window_days = st.sidebar.number_input(
    "未来观察窗口K天",
    min_value=1,
    max_value=60,
    value=10
)
future_window_days = int(future_window_days)

waiting_sensitivity = st.sidebar.slider(
    "等待更好车敏感度",
    min_value=0.0,
    max_value=2.0,
    value=0.50,
    step=0.05
)

fomo_sensitivity = st.sidebar.slider(
    "错失焦虑敏感度",
    min_value=0.0,
    max_value=2.0,
    value=0.40,
    step=0.05
)

choice_dilution_strength = st.sidebar.slider(
    "多车选择稀释强度",
    min_value=0.0,
    max_value=1.0,
    value=0.08,
    step=0.01
)

waiting_penalty_floor = st.sidebar.slider(
    "等待惩罚系数下限",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05
)

st.sidebar.header("档位参数")

tier_car_count = {}
tier_refresh_weight = {}
tier_attractiveness = {}
tier_waiting_penalty_strength = {}

for tier in tiers:
    with st.sidebar.expander(f"{tier} 参数", expanded=False):
        tier_car_count[tier] = st.number_input(
            f"{tier} 车辆数量",
            min_value=1,
            max_value=500,
            value=default_car_count[tier],
            key=f"{tier}_car_count"
        )
        tier_car_count[tier] = int(tier_car_count[tier])

        tier_refresh_weight[tier] = st.number_input(
            f"{tier} 刷新权重",
            min_value=0.0,
            max_value=1000.0,
            value=default_refresh_weight[tier],
            step=1.0,
            key=f"{tier}_refresh_weight"
        )

        tier_attractiveness[tier] = st.number_input(
            f"{tier} 基础吸引力系数",
            min_value=0.0,
            max_value=10.0,
            value=default_attractiveness[tier],
            step=0.1,
            key=f"{tier}_attractiveness"
        )

        tier_waiting_penalty_strength[tier] = st.number_input(
            f"{tier} 等待惩罚强度",
            min_value=0.0,
            max_value=5.0,
            value=default_tier_waiting_penalty_strength[tier],
            step=0.05,
            key=f"{tier}_waiting_penalty_strength"
        )

run = st.sidebar.button("开始模拟", type="primary", use_container_width=True)


# =========================
# 参数计算
# =========================

total_weight = sum(tier_refresh_weight.values())

if total_weight <= 0:
    st.error("所有档位刷新权重之和不能为 0。")
    st.stop()

tier_refresh_prob = {
    tier: tier_refresh_weight[tier] / total_weight
    for tier in tiers
}

high_value_prob = tier_refresh_prob["T1"] + tier_refresh_prob["T2"]

p_high_cycle = 1 - (1 - high_value_prob) ** cars_per_cycle

future_cycle_count = future_window_days / cycle_days
future_high_value_prob = 1 - (1 - p_high_cycle) ** future_cycle_count

choice_dilution_factor = 1 / (1 + choice_dilution_strength * (cars_per_cycle - 1))

fomo_factor = 1 + (
    fomo_sensitivity
    * np.log(1 + cycle_days)
    / np.log(11)
    / np.sqrt(cars_per_cycle)
)

tier_waiting_penalty_factor = {}

for tier in tiers:
    raw_penalty = 1 - (
        waiting_sensitivity
        * future_high_value_prob
        * tier_waiting_penalty_strength[tier]
    )
    tier_waiting_penalty_factor[tier] = max(waiting_penalty_floor, raw_penalty)

tier_final_attractiveness = {
    tier: (
        tier_attractiveness[tier]
        * choice_dilution_factor
        * fomo_factor
        * tier_waiting_penalty_factor[tier]
    )
    for tier in tiers
}

segment_player_counts = allocate_counts(players, segment_weights, segments)

small_r_avg_pay = segment_avg_total_pay["小R"]

segment_ability_coef = {
    seg: (segment_avg_total_pay[seg] / small_r_avg_pay) ** ability_alpha
    for seg in segments
}

segment_budget_per_period = {
    seg: small_r_budget_per_period * segment_ability_coef[seg]
    for seg in segments
}

segment_daily_income = {
    seg: segment_budget_per_period[seg] / fixed_period_days
    for seg in segments
}

segment_initial_fund = {
    seg: segment_budget_per_period[seg] * initial_fund_ratio
    for seg in segments
}

segment_price_sigma = {
    seg: price * segment_price_sigma_coef[seg]
    for seg in segments
}

segment_fund_sigma = {
    seg: price * segment_fund_sigma_coef[seg]
    for seg in segments
}

segment_hesitation_cost = {
    seg: price * segment_hesitation_coef[seg]
    for seg in segments
}


# =========================
# 公式展示
# =========================

st.subheader("模型内部公式")

st.markdown("""
### 1. 玩家资金

资金每天增长，不是在周期第一天一次性发放。

每个玩家在刷新周期内随机一天打开商店，打开当天根据当前资金判断购买。

### 2. 车辆最终吸引力

车辆最终吸引力 = 档位基础吸引力 × 选择稀释系数 × 错失焦虑系数 × 等待惩罚系数

### 3. 心理价位

心理价位 = 基准心理价位 × 车辆最终吸引力 × 玩家层级心理价值系数

### 4. 价格接受概率

价格接受概率 = sigmoid((心理价位 - 实际价格) / 价格σ)

### 5. 资金承受概率

资金承受概率 = sigmoid((当前资金 - 实际价格 - 犹豫成本) / 资金σ)

### 6. 最终购买概率

最终购买概率 = 价格接受概率 × 资金承受概率

### 7. 多车竞争逻辑

玩家每周期刷到多辆车后：

1. 按车辆吸引力从高到低尝试购买
2. 吸引力相同则随机排序
3. 根据双Sigmoid概率判断是否购买
4. 购买后扣除单车价格
5. 同一玩家同周期不会重复购买同一辆车
6. 周期结束结算打开后剩余天数的资金增长
""")


# =========================
# 参数展示
# =========================

st.subheader("刷新心理参数")

psychology_df = pd.DataFrame([
    {"参数": "高价值车概率(T1+T2)", "数值": high_value_prob},
    {"参数": "单周期刷到高价值车概率", "数值": p_high_cycle},
    {"参数": "未来K天刷到高价值车概率", "数值": future_high_value_prob},
    {"参数": "选择稀释系数", "数值": choice_dilution_factor},
    {"参数": "错失焦虑系数", "数值": fomo_factor},
    {"参数": "未来观察窗口K天", "数值": future_window_days},
    {"参数": "等待更好车敏感度", "数值": waiting_sensitivity},
    {"参数": "错失焦虑敏感度", "数值": fomo_sensitivity},
    {"参数": "多车选择稀释强度", "数值": choice_dilution_strength},
    {"参数": "等待惩罚系数下限", "数值": waiting_penalty_floor},
])

display_psychology_df = psychology_df.copy()
display_psychology_df["数值"] = display_psychology_df.apply(
    lambda row: f"{float(row['数值']):.2%}" if "概率" in row["参数"] else f"{float(row['数值']):,.4f}",
    axis=1
)

st.dataframe(display_psychology_df, use_container_width=True)

st.subheader("玩家分层参数")

segment_summary = []

for seg in segments:
    segment_summary.append({
        "玩家层级": seg,
        "分层权重": segment_weights[seg],
        "本周期玩家数": segment_player_counts[seg],
        "平均累计付费": segment_avg_total_pay[seg],
        "付费能力系数": segment_ability_coef[seg],
        "固定周期车辆预算": segment_budget_per_period[seg],
        "每日新增资金": segment_daily_income[seg],
        "初始资金": segment_initial_fund[seg],
        "心理价值系数": segment_value_coef[seg],
        "价格σ系数": segment_price_sigma_coef[seg],
        "价格σ实际值": segment_price_sigma[seg],
        "资金σ系数": segment_fund_sigma_coef[seg],
        "资金σ实际值": segment_fund_sigma[seg],
        "犹豫成本系数": segment_hesitation_coef[seg],
        "犹豫成本": segment_hesitation_cost[seg],
    })

segment_summary_df = pd.DataFrame(segment_summary)

display_segment_summary = segment_summary_df.copy()
display_segment_summary["分层权重"] = display_segment_summary["分层权重"].map(lambda x: f"{x:.2%}")

for col in [
    "平均累计付费",
    "付费能力系数",
    "固定周期车辆预算",
    "每日新增资金",
    "初始资金",
    "心理价值系数",
    "价格σ系数",
    "价格σ实际值",
    "资金σ系数",
    "资金σ实际值",
    "犹豫成本系数",
    "犹豫成本",
]:
    display_segment_summary[col] = display_segment_summary[col].map(lambda x: f"{x:,.2f}")

st.dataframe(display_segment_summary, use_container_width=True)

st.subheader("档位参数")

tier_summary = []

for tier in tiers:
    tier_summary.append({
        "档位": tier,
        "车辆数量": tier_car_count[tier],
        "刷新权重": tier_refresh_weight[tier],
        "实际刷新概率": tier_refresh_prob[tier],
        "基础吸引力系数": tier_attractiveness[tier],
        "选择稀释系数": choice_dilution_factor,
        "错失焦虑系数": fomo_factor,
        "等待惩罚强度": tier_waiting_penalty_strength[tier],
        "等待惩罚系数": tier_waiting_penalty_factor[tier],
        "最终吸引力": tier_final_attractiveness[tier],
    })

tier_summary_df = pd.DataFrame(tier_summary)

display_tier_summary = tier_summary_df.copy()
display_tier_summary["实际刷新概率"] = display_tier_summary["实际刷新概率"].map(lambda x: f"{x:.2%}")

for col in [
    "基础吸引力系数",
    "选择稀释系数",
    "错失焦虑系数",
    "等待惩罚强度",
    "等待惩罚系数",
    "最终吸引力",
]:
    display_tier_summary[col] = display_tier_summary[col].map(lambda x: f"{x:,.4f}")

st.dataframe(display_tier_summary, use_container_width=True)

st.subheader("当前基础参数")

param_df = pd.DataFrame([
    {"参数": "刷新周期", "数值": cycle_days},
    {"参数": "该周期去重登录小R及以上玩家数", "数值": players},
    {"参数": "模拟周期数", "数值": simulate_cycles},
    {"参数": "总模拟天数", "数值": total_days},
    {"参数": "每周期每人刷车数", "数值": cars_per_cycle},
    {"参数": "单车价格", "数值": price},
    {"参数": "固定资金周期天数", "数值": fixed_period_days},
    {"参数": "小R固定周期车辆预算", "数值": small_r_budget_per_period},
    {"参数": "付费能力压缩指数α", "数值": ability_alpha},
    {"参数": "初始资金比例", "数值": initial_fund_ratio},
    {"参数": "允许资金为负", "数值": allow_negative_funds},
    {"参数": "基准心理价位", "数值": base_psychological_value},
])

st.dataframe(param_df, use_container_width=True)


# =========================
# 模拟逻辑
# =========================

if run:
    rng = np.random.default_rng(random_seed)

    car_rows = []
    tier_index_map = {tier: idx for idx, tier in enumerate(tiers)}

    car_global_index = 0
    tier_car_start_index = {}

    for tier in tiers:
        tier_car_start_index[tier] = car_global_index
        for i in range(1, tier_car_count[tier] + 1):
            car_rows.append({
                "car_id": car_global_index,
                "car": f"{tier}_Car_{i}",
                "tier": tier
            })
            car_global_index += 1

    car_df = pd.DataFrame(car_rows)
    total_cars = len(car_df)

    tier_prob_array = np.array([tier_refresh_prob[tier] for tier in tiers])
    tier_attr_array = np.array([tier_final_attractiveness[tier] for tier in tiers])

    player_funds = {}

    for seg in segments:
        count = segment_player_counts[seg]
        base_initial = segment_initial_fund[seg]

        low = base_initial * 0.5
        high = base_initial * 1.5

        funds = rng.uniform(low=low, high=high, size=count)
        player_funds[seg] = funds

    cycle_records = []
    segment_records = []
    tier_records = []
    car_records = []

    for cycle in range(1, simulate_cycles + 1):
        cycle_exposure = 0
        cycle_sales = 0
        cycle_revenue = 0

        for seg in segments:
            player_count = segment_player_counts[seg]

            if player_count <= 0:
                continue

            funds_start_cycle = player_funds[seg]

            open_day = rng.integers(
                low=1,
                high=cycle_days + 1,
                size=player_count
            )

            funds_at_open = funds_start_cycle + segment_daily_income[seg] * open_day
            funds = funds_at_open.copy()

            tier_indices = rng.choice(
                len(tiers),
                size=(player_count, cars_per_cycle),
                p=tier_prob_array
            )

            car_ids = np.empty_like(tier_indices)

            for tier in tiers:
                tier_idx = tier_index_map[tier]
                mask = tier_indices == tier_idx
                mask_count = int(mask.sum())

                if mask_count > 0:
                    start_idx = tier_car_start_index[tier]
                    count_in_tier = tier_car_count[tier]
                    car_ids[mask] = start_idx + rng.integers(
                        low=0,
                        high=count_in_tier,
                        size=mask_count
                    )

            exposure_by_tier = np.bincount(
                tier_indices.ravel(),
                minlength=len(tiers)
            )

            exposure_by_car = np.bincount(
                car_ids.ravel(),
                minlength=total_cars
            )

            attractiveness_matrix = tier_attr_array[tier_indices]

            psychological_value = (
                base_psychological_value
                * attractiveness_matrix
                * segment_value_coef[seg]
            )

            price_sigma = max(segment_price_sigma[seg], 1e-9)
            fund_sigma = max(segment_fund_sigma[seg], 1e-9)
            hesitation_cost = segment_hesitation_cost[seg]

            price_accept_prob = sigmoid((psychological_value - price) / price_sigma)

            tie_noise = rng.random(size=attractiveness_matrix.shape) * 1e-6
            purchase_order = np.argsort(
                -(attractiveness_matrix + tie_noise),
                axis=1
            )

            sales_by_car = np.zeros(total_cars, dtype=np.int64)
            purchased_ids_by_player = np.full(
                (player_count, cars_per_cycle),
                -1,
                dtype=np.int64
            )

            row_index = np.arange(player_count)

            for rank in range(cars_per_cycle):
                col_index = purchase_order[:, rank]

                candidate_car_ids = car_ids[row_index, col_index]
                candidate_price_accept_prob = price_accept_prob[row_index, col_index]

                fund_accept_prob = sigmoid(
                    (funds - price - hesitation_cost) / fund_sigma
                )

                candidate_purchase_prob = candidate_price_accept_prob * fund_accept_prob

                if not allow_negative_funds:
                    candidate_purchase_prob = np.where(funds >= price, candidate_purchase_prob, 0)

                if rank > 0:
                    already_bought_same_car = (
                        purchased_ids_by_player[:, :rank] == candidate_car_ids[:, None]
                    ).any(axis=1)
                else:
                    already_bought_same_car = np.zeros(player_count, dtype=bool)

                random_roll = rng.random(player_count)

                can_buy = (
                    (~already_bought_same_car)
                    & (random_roll < candidate_purchase_prob)
                )

                if np.any(can_buy):
                    bought_car_ids = candidate_car_ids[can_buy]
                    np.add.at(sales_by_car, bought_car_ids, 1)

                    funds[can_buy] -= price
                    purchased_ids_by_player[can_buy, rank] = bought_car_ids

            remaining_days_after_open = cycle_days - open_day

            funds_end_cycle = funds + segment_daily_income[seg] * remaining_days_after_open
            player_funds[seg] = funds_end_cycle

            sales_by_tier = np.zeros(len(tiers), dtype=np.int64)

            for tier in tiers:
                tier_idx = tier_index_map[tier]
                start_idx = tier_car_start_index[tier]
                end_idx = start_idx + tier_car_count[tier]
                sales_by_tier[tier_idx] = sales_by_car[start_idx:end_idx].sum()

            segment_exposure = int(exposure_by_tier.sum())
            segment_sales = int(sales_by_tier.sum())
            segment_revenue = segment_sales * price

            cycle_exposure += segment_exposure
            cycle_sales += segment_sales
            cycle_revenue += segment_revenue

            segment_records.append({
                "cycle": cycle,
                "segment": seg,
                "players": player_count,
                "avg_open_day": float(np.mean(open_day)) if len(open_day) > 0 else 0,
                "exposure": segment_exposure,
                "sales": segment_sales,
                "revenue": segment_revenue,
                "avg_funds_at_open": float(np.mean(funds_at_open)) if len(funds_at_open) > 0 else 0,
                "avg_remaining_funds": float(np.mean(funds_end_cycle)) if len(funds_end_cycle) > 0 else 0,
            })

            for tier in tiers:
                tier_idx = tier_index_map[tier]

                tier_records.append({
                    "cycle": cycle,
                    "segment": seg,
                    "tier": tier,
                    "exposure": int(exposure_by_tier[tier_idx]),
                    "sales": int(sales_by_tier[tier_idx]),
                    "revenue": int(sales_by_tier[tier_idx]) * price,
                })

            for _, car_row in car_df.iterrows():
                car_id = int(car_row["car_id"])
                exposure = int(exposure_by_car[car_id])
                sales = int(sales_by_car[car_id])

                if exposure > 0 or sales > 0:
                    car_records.append({
                        "cycle": cycle,
                        "segment": seg,
                        "car": car_row["car"],
                        "tier": car_row["tier"],
                        "exposure": exposure,
                        "sales": sales,
                        "revenue": sales * price,
                    })

        cycle_records.append({
            "cycle": cycle,
            "start_day": (cycle - 1) * cycle_days + 1,
            "end_day": cycle * cycle_days,
            "exposure": cycle_exposure,
            "sales": cycle_sales,
            "revenue": cycle_revenue,
            "daily_avg_exposure": cycle_exposure / cycle_days,
            "daily_avg_sales": cycle_sales / cycle_days,
            "daily_avg_revenue": cycle_revenue / cycle_days,
        })

    cycle_df = pd.DataFrame(cycle_records)
    segment_df = pd.DataFrame(segment_records)
    tier_df = pd.DataFrame(tier_records)
    car_result_df = pd.DataFrame(car_records)

    total_exposure = cycle_df["exposure"].sum()
    total_sales = cycle_df["sales"].sum()
    total_revenue = cycle_df["revenue"].sum()

    st.subheader("核心模拟结果")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("总曝光次数", f"{int(total_exposure):,}")
    col2.metric("总销量", f"{int(total_sales):,}")
    col3.metric("总流水", f"{total_revenue:,.2f}")
    col4.metric("日均流水", f"{total_revenue / total_days:,.2f}")

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("单周期平均曝光", f"{cycle_df['exposure'].mean():,.2f}")
    col6.metric("单周期平均销量", f"{cycle_df['sales'].mean():,.2f}")
    col7.metric("单周期平均流水", f"{cycle_df['revenue'].mean():,.2f}")
    col8.metric("日均销量", f"{total_sales / total_days:,.2f}")

    st.subheader("周期结果")
    st.dataframe(cycle_df, use_container_width=True)

    cycle_sales_chart = alt.Chart(cycle_df).mark_line(point=True).encode(
        x=alt.X("cycle:O", title="周期"),
        y=alt.Y("sales:Q", title="周期销量"),
        tooltip=[
            "cycle",
            "start_day",
            "end_day",
            "exposure",
            "sales",
            "revenue",
            "daily_avg_sales",
            "daily_avg_revenue",
        ]
    ).properties(title="各周期车辆销量")

    st.altair_chart(cycle_sales_chart, use_container_width=True)

    cycle_revenue_chart = alt.Chart(cycle_df).mark_line(point=True).encode(
        x=alt.X("cycle:O", title="周期"),
        y=alt.Y("revenue:Q", title="周期流水"),
        tooltip=[
            "cycle",
            "start_day",
            "end_day",
            "exposure",
            "sales",
            "revenue",
            "daily_avg_sales",
            "daily_avg_revenue",
        ]
    ).properties(title="各周期车辆流水")

    st.altair_chart(cycle_revenue_chart, use_container_width=True)

    st.subheader("玩家层级累计结果")

    segment_total_df = segment_df.groupby("segment", as_index=False).agg(
        players=("players", "max"),
        avg_open_day=("avg_open_day", "mean"),
        exposure=("exposure", "sum"),
        sales=("sales", "sum"),
        revenue=("revenue", "sum"),
        avg_funds_at_open=("avg_funds_at_open", "mean"),
        avg_remaining_funds=("avg_remaining_funds", "mean"),
    )

    segment_total_df["daily_avg_sales"] = segment_total_df["sales"] / total_days
    segment_total_df["daily_avg_revenue"] = segment_total_df["revenue"] / total_days
    segment_total_df["conversion_rate"] = (
        segment_total_df["sales"]
        / segment_total_df["exposure"].replace(0, np.nan)
    )

    display_segment_total_df = segment_total_df.copy()
    display_segment_total_df["conversion_rate"] = display_segment_total_df["conversion_rate"].map(
        lambda x: "0.00%" if pd.isna(x) else f"{x:.2%}"
    )

    st.dataframe(display_segment_total_df, use_container_width=True)

    segment_revenue_chart = alt.Chart(segment_total_df).mark_bar().encode(
        x=alt.X("segment:N", title="玩家层级"),
        y=alt.Y("revenue:Q", title="累计流水"),
        tooltip=[
            "segment",
            "players",
            "avg_open_day",
            "exposure",
            "sales",
            "revenue",
            "daily_avg_sales",
            "daily_avg_revenue",
            "avg_funds_at_open",
            "avg_remaining_funds",
        ]
    ).properties(title="各玩家层级累计流水")

    st.altair_chart(segment_revenue_chart, use_container_width=True)

    st.subheader("档位累计结果")

    tier_total_df = tier_df.groupby("tier", as_index=False).agg(
        exposure=("exposure", "sum"),
        sales=("sales", "sum"),
        revenue=("revenue", "sum"),
    )

    tier_total_df["daily_avg_sales"] = tier_total_df["sales"] / total_days
    tier_total_df["daily_avg_revenue"] = tier_total_df["revenue"] / total_days
    tier_total_df["conversion_rate"] = (
        tier_total_df["sales"]
        / tier_total_df["exposure"].replace(0, np.nan)
    )

    display_tier_total_df = tier_total_df.copy()
    display_tier_total_df["conversion_rate"] = display_tier_total_df["conversion_rate"].map(
        lambda x: "0.00%" if pd.isna(x) else f"{x:.2%}"
    )

    st.dataframe(display_tier_total_df, use_container_width=True)

    tier_sales_chart = alt.Chart(tier_total_df).mark_bar().encode(
        x=alt.X("tier:N", title="档位"),
        y=alt.Y("sales:Q", title="累计销量"),
        tooltip=[
            "tier",
            "exposure",
            "sales",
            "revenue",
            "daily_avg_sales",
            "daily_avg_revenue",
        ]
    ).properties(title="各档位累计销量")

    st.altair_chart(tier_sales_chart, use_container_width=True)

    st.subheader("车辆累计结果")

    if len(car_result_df) > 0:
        car_total_df = car_result_df.groupby(["car", "tier"], as_index=False).agg(
            exposure=("exposure", "sum"),
            sales=("sales", "sum"),
            revenue=("revenue", "sum"),
        ).sort_values("sales", ascending=False)

        car_total_df["daily_avg_sales"] = car_total_df["sales"] / total_days
        car_total_df["daily_avg_revenue"] = car_total_df["revenue"] / total_days
        car_total_df["conversion_rate"] = (
            car_total_df["sales"]
            / car_total_df["exposure"].replace(0, np.nan)
        )

        display_car_total_df = car_total_df.copy()
        display_car_total_df["conversion_rate"] = display_car_total_df["conversion_rate"].map(
            lambda x: "0.00%" if pd.isna(x) else f"{x:.2%}"
        )

        st.dataframe(display_car_total_df, use_container_width=True)

        car_sales_chart = alt.Chart(car_total_df.head(30)).mark_bar().encode(
            x=alt.X("car:N", sort="-y", title="车辆"),
            y=alt.Y("sales:Q", title="累计销量"),
            color=alt.Color("tier:N", title="档位"),
            tooltip=[
                "car",
                "tier",
                "exposure",
                "sales",
                "revenue",
                "daily_avg_sales",
                "daily_avg_revenue",
            ]
        ).properties(title="车辆累计销量 TOP 30")

        st.altair_chart(car_sales_chart, use_container_width=True)

        st.download_button(
            label="下载车辆结果 CSV",
            data=car_total_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="car_shop_sim_car_result.csv",
            mime="text/csv",
        )

    st.download_button(
        label="下载周期结果 CSV",
        data=cycle_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="car_shop_sim_cycle_result.csv",
        mime="text/csv",
    )

    st.download_button(
        label="下载玩家层级结果 CSV",
        data=segment_total_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="car_shop_sim_segment_result.csv",
        mime="text/csv",
    )
