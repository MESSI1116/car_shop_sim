import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

st.set_page_config(page_title="每日车辆商店销量模拟器", layout="wide")

st.title("每日车辆商店销量模拟器")

# n天窗口下，小R及以上平均去重登录玩家数
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
    max_value=1000,
    value=10
)

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

base_p = st.sidebar.slider(
    "基础单车购买概率 p（周期内，T3基准）",
    min_value=0.0,
    max_value=1.0,
    value=0.01,
    step=0.001
)

price = st.sidebar.number_input(
    "单车价格",
    min_value=0.0,
    value=68.0
)

st.sidebar.header("档位参数")

tier_car_count = {}
tier_refresh_weight = {}
tier_attractiveness = {}

for tier in tiers:
    with st.sidebar.expander(f"{tier} 参数", expanded=False):
        tier_car_count[tier] = st.number_input(
            f"{tier} 车辆数量",
            min_value=1,
            max_value=500,
            value=default_car_count[tier],
            key=f"{tier}_car_count"
        )

        tier_refresh_weight[tier] = st.number_input(
            f"{tier} 刷新权重",
            min_value=0.0,
            max_value=1000.0,
            value=default_refresh_weight[tier],
            step=1.0,
            key=f"{tier}_refresh_weight"
        )

        tier_attractiveness[tier] = st.number_input(
            f"{tier} 吸引力系数",
            min_value=0.0,
            max_value=10.0,
            value=default_attractiveness[tier],
            step=0.1,
            key=f"{tier}_attractiveness"
        )

total_weight = sum(tier_refresh_weight.values())

if total_weight <= 0:
    st.error("所有档位刷新权重之和不能为 0。")
    st.stop()

# =========================
# 模型参数计算
# =========================

tier_refresh_prob = {
    tier: tier_refresh_weight[tier] / total_weight
    for tier in tiers
}

cycle_factor = 0.7 + 0.6 * (cycle_days - 1) / 9

tier_final_p = {
    tier: min(base_p * tier_attractiveness[tier] * cycle_factor, 0.95)
    for tier in tiers
}

exposure_per_cycle = players * cars_per_cycle

tier_summary = []

for tier in tiers:
    expected_exposure_per_cycle = exposure_per_cycle * tier_refresh_prob[tier]
    expected_sales_per_cycle = expected_exposure_per_cycle * tier_final_p[tier]
    expected_revenue_per_cycle = expected_sales_per_cycle * price

    tier_summary.append({
        "档位": tier,
        "车辆数量": tier_car_count[tier],
        "刷新权重": tier_refresh_weight[tier],
        "实际刷新概率": tier_refresh_prob[tier],
        "吸引力系数": tier_attractiveness[tier],
        "最终单车购买概率": tier_final_p[tier],
        "单周期期望曝光": expected_exposure_per_cycle,
        "单周期期望销量": expected_sales_per_cycle,
        "单周期期望流水": expected_revenue_per_cycle,
        "日均期望曝光": expected_exposure_per_cycle / cycle_days,
        "日均期望销量": expected_sales_per_cycle / cycle_days,
        "日均期望流水": expected_revenue_per_cycle / cycle_days,
    })

tier_summary_df = pd.DataFrame(tier_summary)

# 开始模拟按钮放在侧边栏，方便找到
run = st.sidebar.button("开始模拟", type="primary", use_container_width=True)

# =========================
# 公式展示
# =========================

st.subheader("模型内部公式")

st.markdown("""
### 1. 玩家数
玩家数 = 当前刷新周期 n 天内，去重登录的小R及以上玩家数

### 2. 单周期曝光
单周期总曝光 = n天去重登录玩家数 × 每个玩家每周期刷到车辆数

### 3. 档位实际刷新概率
某档位实际刷新概率 = 该档位刷新权重 / 所有档位刷新权重之和

### 4. 刷新周期系数
周期系数 = 0.7 + 0.6 × (刷新周期 - 1) / 9

### 5. 最终单车购买概率
最终单车购买概率 = 基础购买概率p × 档位吸引力系数 × 周期系数

最终单车购买概率最高不超过 95%

### 6. 某档位单周期期望销量
某档位单周期期望销量 = 单周期总曝光 × 该档位实际刷新概率 × 该档位最终单车购买概率

### 7. 日均销量
日均销量 = 单周期销量 / 刷新周期天数

### 8. 流水
周期流水 = 周期销量 × 单车价格  
日均流水 = 周期流水 / 刷新周期天数
""")

# =========================
# 期望结果展示
# =========================

st.subheader("档位参数与期望结果")

display_tier_summary = tier_summary_df.copy()

for col in ["实际刷新概率", "最终单车购买概率"]:
    display_tier_summary[col] = display_tier_summary[col].map(lambda x: f"{x:.2%}")

for col in [
    "单周期期望曝光",
    "单周期期望销量",
    "单周期期望流水",
    "日均期望曝光",
    "日均期望销量",
    "日均期望流水",
]:
    display_tier_summary[col] = display_tier_summary[col].map(lambda x: f"{x:,.2f}")

st.dataframe(display_tier_summary, use_container_width=True)

st.subheader("当前基础参数")

param_df = pd.DataFrame([
    {"参数": "刷新周期", "数值": cycle_days},
    {"参数": "该周期去重登录小R及以上玩家数", "数值": players},
    {"参数": "模拟周期数", "数值": simulate_cycles},
    {"参数": "总模拟天数", "数值": total_days},
    {"参数": "每周期每人刷车数", "数值": cars_per_cycle},
    {"参数": "基础单车购买概率p", "数值": base_p},
    {"参数": "周期系数", "数值": cycle_factor},
    {"参数": "单车价格", "数值": price},
])

st.dataframe(param_df, use_container_width=True)

# =========================
# 模拟逻辑
# =========================

if run:
    car_rows = []

    for tier in tiers:
        for i in range(1, tier_car_count[tier] + 1):
            car_rows.append({
                "car": f"{tier}_Car_{i}",
                "tier": tier
            })

    car_df = pd.DataFrame(car_rows)

    results = []

    for cycle in range(1, simulate_cycles + 1):
        total_slots = exposure_per_cycle

        tier_exposures = np.random.multinomial(
            total_slots,
            [tier_refresh_prob[tier] for tier in tiers]
        )

        for tier, exposure_count in zip(tiers, tier_exposures):
            cars_in_tier = car_df[car_df["tier"] == tier]["car"].tolist()
            car_num = len(cars_in_tier)

            if exposure_count <= 0:
                continue

            car_exposures = np.random.multinomial(
                exposure_count,
                [1 / car_num] * car_num
            )

            for car, exposure in zip(cars_in_tier, car_exposures):
                sales = np.random.binomial(exposure, tier_final_p[tier])
                revenue = sales * price

                results.append({
                    "cycle": cycle,
                    "start_day": (cycle - 1) * cycle_days + 1,
                    "end_day": cycle * cycle_days,
                    "car": car,
                    "tier": tier,
                    "exposure": exposure,
                    "sales": sales,
                    "revenue": revenue,
                })

    df = pd.DataFrame(results)

    cycle_total = df.groupby("cycle", as_index=False).agg(
        exposure=("exposure", "sum"),
        sales=("sales", "sum"),
        revenue=("revenue", "sum")
    )

    cycle_total["daily_avg_exposure"] = cycle_total["exposure"] / cycle_days
    cycle_total["daily_avg_sales"] = cycle_total["sales"] / cycle_days
    cycle_total["daily_avg_revenue"] = cycle_total["revenue"] / cycle_days

    tier_total = df.groupby("tier", as_index=False).agg(
        exposure=("exposure", "sum"),
        sales=("sales", "sum"),
        revenue=("revenue", "sum")
    )

    tier_total["daily_avg_exposure"] = tier_total["exposure"] / total_days
    tier_total["daily_avg_sales"] = tier_total["sales"] / total_days
    tier_total["daily_avg_revenue"] = tier_total["revenue"] / total_days
    tier_total["conversion_rate"] = tier_total["sales"] / tier_total["exposure"].replace(0, np.nan)

    car_total = df.groupby(["car", "tier"], as_index=False).agg(
        exposure=("exposure", "sum"),
        sales=("sales", "sum"),
        revenue=("revenue", "sum")
    ).sort_values("sales", ascending=False)

    car_total["daily_avg_exposure"] = car_total["exposure"] / total_days
    car_total["daily_avg_sales"] = car_total["sales"] / total_days
    car_total["daily_avg_revenue"] = car_total["revenue"] / total_days
    car_total["conversion_rate"] = car_total["sales"] / car_total["exposure"].replace(0, np.nan)

    st.subheader("核心模拟结果")

    total_exposure = cycle_total["exposure"].sum()
    total_sales = cycle_total["sales"].sum()
    total_revenue = cycle_total["revenue"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("总曝光次数", f"{int(total_exposure):,}")
    col2.metric("总销量", f"{int(total_sales):,}")
    col3.metric("总流水", f"{total_revenue:,.2f}")
    col4.metric("日均流水", f"{total_revenue / total_days:,.2f}")

    col5, col6, col7, col8 = st.columns(4)

    col5.metric("单周期平均曝光", f"{cycle_total['exposure'].mean():,.2f}")
    col6.metric("单周期平均销量", f"{cycle_total['sales'].mean():,.2f}")
    col7.metric("单周期平均流水", f"{cycle_total['revenue'].mean():,.2f}")
    col8.metric("日均销量", f"{total_sales / total_days:,.2f}")

    st.subheader("周期结果")
    st.dataframe(cycle_total, use_container_width=True)

    cycle_sales_chart = alt.Chart(cycle_total).mark_line(point=True).encode(
        x=alt.X("cycle:O", title="周期"),
        y=alt.Y("sales:Q", title="周期销量"),
        tooltip=["cycle", "exposure", "sales", "revenue", "daily_avg_sales", "daily_avg_revenue"]
    ).properties(title="各周期车辆销量")

    st.altair_chart(cycle_sales_chart, use_container_width=True)

    cycle_revenue_chart = alt.Chart(cycle_total).mark_line(point=True).encode(
        x=alt.X("cycle:O", title="周期"),
        y=alt.Y("revenue:Q", title="周期流水"),
        tooltip=["cycle", "exposure", "sales", "revenue", "daily_avg_sales", "daily_avg_revenue"]
    ).properties(title="各周期车辆流水")

    st.altair_chart(cycle_revenue_chart, use_container_width=True)

    st.subheader("档位累计结果")

    display_tier_total = tier_total.copy()
    display_tier_total["conversion_rate"] = display_tier_total["conversion_rate"].map(
        lambda x: "0.00%" if pd.isna(x) else f"{x:.2%}"
    )

    st.dataframe(display_tier_total, use_container_width=True)

    tier_sales_chart = alt.Chart(tier_total).mark_bar().encode(
        x=alt.X("tier:N", title="档位"),
        y=alt.Y("sales:Q", title="累计销量"),
        tooltip=["tier", "exposure", "sales", "revenue", "daily_avg_sales", "daily_avg_revenue"]
    ).properties(title="各档位累计销量")

    st.altair_chart(tier_sales_chart, use_container_width=True)

    st.subheader("车辆累计结果")

    display_car_total = car_total.copy()
    display_car_total["conversion_rate"] = display_car_total["conversion_rate"].map(
        lambda x: "0.00%" if pd.isna(x) else f"{x:.2%}"
    )

    st.dataframe(display_car_total, use_container_width=True)

    car_sales_chart = alt.Chart(car_total.head(30)).mark_bar().encode(
        x=alt.X("car:N", sort="-y", title="车辆"),
        y=alt.Y("sales:Q", title="累计销量"),
        color=alt.Color("tier:N", title="档位"),
        tooltip=["car", "tier", "exposure", "sales", "revenue", "daily_avg_sales", "daily_avg_revenue"]
    ).properties(title="车辆累计销量 TOP 30")

    st.altair_chart(car_sales_chart, use_container_width=True)
