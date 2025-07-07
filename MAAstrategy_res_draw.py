import pandas as pd
import matplotlib.pyplot as plt
import os

# 加载CSV数据
csv_path = "./comprehensive_analysis_reports/MAA_策略分析/MAA_vs_Others_Stats_All.csv"
df = pd.read_csv(csv_path)

# 定义指标分组（用于分面图）
metric_groups = {
    "Return-related": [
        'total_return', 'annual_return', 'return_mean', 'return_std', 'return_skew', 'return_kurt', 
        'positive_days_pct', 'max_daily_gain', 'max_daily_loss'
    ],
    "Risk-related": [
        'max_drawdown_pct', 'avg_drawdown', 'drawdown_days', 'var_95', 'var_99', 'volatility'
    ],
    "Risk-Adjusted": [
        'sharpe_ratio', 'calmar_ratio'
    ],
    "Portfolio Structure": [
        'avg_concentration', 'weight_volatility', 'avg_active_assets', 'max_weight',
        'avg_weight', 'weight_dispersion', 'max_individual_weight', 'active_asset_ratio',
        'avg_long_short_ratio'
    ],
    "Capital": [
        'initial_capital', 'final_capital'
    ],
    "Model Evaluation": [
        'test_mse', 'test_accuracy', 'investment_accuracy'
    ],
    "Meta": [
        'backtest_days'
    ]
}

# 专业短名称（用于X轴显示）
short_names = {
    'total_return': 'TotalRet', 'annual_return': 'AnnRet', 'return_mean': 'RetMean',
    'return_std': 'RetStd', 'return_skew': 'RetSkew', 'return_kurt': 'RetKurt',
    'positive_days_pct': 'PosDays%', 'max_daily_gain': 'MaxGain', 'max_daily_loss': 'MaxLoss',
    'max_drawdown_pct': 'MaxDD%', 'avg_drawdown': 'AvgDD', 'drawdown_days': 'DDD', 
    'var_95': 'VaR95', 'var_99': 'VaR99', 'volatility': 'Vol',
    'sharpe_ratio': 'Sharpe', 'calmar_ratio': 'Calmar',
    'avg_concentration': 'AvgConc', 'weight_volatility': 'WVol', 'avg_active_assets': 'ActAst',
    'max_weight': 'MaxWgt', 'avg_weight': 'AvgWgt', 'weight_dispersion': 'WDisp',
    'max_individual_weight': 'MaxIndWgt', 'active_asset_ratio': 'ActAst%',
    'avg_long_short_ratio': 'Long/Short',
    'initial_capital': 'InitCap', 'final_capital': 'FinalCap',
    'test_mse': 'TestMSE', 'test_accuracy': 'TestAcc', 'investment_accuracy': 'InvestAcc',
    'backtest_days': 'BackDays'
}

# 要展示的统计方式
selected_stats = {
    "min": "min_improvement_pct",
    "q25": "q25_improvement_pct",
    "median": "median_improvement_pct"
}

# 准备数据
df_plot = df[["metric"] + list(selected_stats.values())].copy()
df_plot = df_plot.set_index("metric")
df_plot.columns = list(selected_stats.keys())

# 移除全部为 0 的行
df_plot = df_plot.loc[~(df_plot == 0).all(axis=1)]

# 使用专业术语替代指标名称
df_plot.index = df_plot.index.map(lambda m: short_names.get(m, m))

# 创建保存图像的路径列表
output_paths = []

# 为每一组指标单独绘图
for group_name, metrics in metric_groups.items():
    # 筛选对应的简写指标
    short_metrics = [short_names[m] for m in metrics if short_names.get(m, m) in df_plot.index]
    group_df = df_plot.loc[df_plot.index.isin(short_metrics)]

    if group_df.empty:
        continue

    # 开始绘图
    ax = group_df.plot(kind="bar", figsize=(14, 6), width=0.75)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f"MAA Improvement vs Others – {group_name}")
    plt.ylabel("Improvement (%)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # 保存图像
    dir_path = "./comprehensive_analysis_reports/MAA_策略分析/calculated"
    os.makedirs(dir_path, exist_ok=True)
    output_file = f"{dir_path}/MAA_vs_Others_{group_name.replace(' ', '_')}.png"
    plt.savefig(output_file, dpi=300)
    output_paths.append(output_file)
    plt.close()

# 输出生成的图像路径（可用于后续显示或下载）
print("图像生成完毕，共生成：", len(output_paths), "张图。")
for path in output_paths:
    print(path)
