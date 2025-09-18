# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib

# 1. 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12  # 统一字体大小

# 2. 设置可视化风格
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 3. 数据加载函数
def load_data():
    print("正在加载数据...")
    try:
        actual_df = pd.read_csv('label.csv', 
                              parse_dates=['datetime'],
                              encoding='utf-8')  # 尝试utf-8编码
        predicted_df = pd.read_csv('pred.csv', 
                                 parse_dates=['datetime'],
                                 encoding='utf-8')
        print("数据加载成功！")
        return actual_df, predicted_df
    except UnicodeDecodeError:
        try:
            # 如果utf-8失败，尝试gbk编码
            actual_df = pd.read_csv('actual_returns.csv', 
                                  parse_dates=['datetime'],
                                  encoding='gbk')
            predicted_df = pd.read_csv('predicted_returns.csv', 
                                     parse_dates=['datetime'],
                                     encoding='gbk')
            print("数据加载成功（使用GBK编码）！")
            return actual_df, predicted_df
        except Exception as e:
            print(f"数据加载失败: {e}")
            exit()

# 4. 计算指标函数
def calculate_metrics(actual, predicted):
    mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual, predicted = actual[mask], predicted[mask]
    if len(actual) == 0:
        return {k: np.nan for k in ['MAE', 'MSE', 'RMSE', 'MAPE (%)', 'Direction Accuracy (%)', 'Sample Size']}
    
    abs_actual = np.abs(actual)
    mape_mask = abs_actual > 1e-10
    mape = np.mean(np.abs((actual[mape_mask]-predicted[mape_mask])/abs_actual[mape_mask]))*100 if any(mape_mask) else np.nan
    
    return {
        'MAE': mean_absolute_error(actual, predicted),
        'MSE': mean_squared_error(actual, predicted),
        'RMSE': np.sqrt(mean_squared_error(actual, predicted)),
        'MAPE (%)': mape,
        'Direction Accuracy (%)': np.mean(np.sign(actual)==np.sign(predicted))*100,
        'Sample Size': len(actual)
    }

# 5. 可视化函数
def create_visualizations(df, stock_metrics, monthly_metrics):
    # 预测vs实际散点图
    plt.figure(figsize=(10, 8))
    sns.regplot(x='predicted_return', y='actual_return', data=df, 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.plot([df['actual_return'].min(), df['actual_return'].max()], 
             [df['actual_return'].min(), df['actual_return'].max()], 'k--')
    plt.title('预测收益 vs 实际收益对比', fontsize=15)
    plt.xlabel('预测收益值', fontsize=12)
    plt.ylabel('实际收益值', fontsize=12)
    plt.tight_layout()
    plt.savefig('result/pred_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 误差分布图
    plt.figure(figsize=(10, 6))
    sns.histplot(df['error'], bins=50, kde=True)
    plt.title('预测误差分布情况', fontsize=15)
    plt.xlabel('预测误差（实际-预测）', fontsize=12)
    plt.ylabel('出现频次', fontsize=12)
    plt.tight_layout()
    plt.savefig('result/error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 方向准确性按股票
    min_samples = 10
    valid_stocks = stock_metrics[stock_metrics['Sample Size'] >= min_samples].index
    if len(valid_stocks) > 0:
        plt.figure(figsize=(12, 6))
        stock_metrics.loc[valid_stocks, 'Direction Accuracy (%)'].sort_values().plot(kind='bar')
        plt.title(f'各股票预测方向准确性 (样本量≥{min_samples})', fontsize=15)
        plt.xlabel('股票代码', fontsize=12)
        plt.ylabel('方向准确性 (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('result/direction_accuracy_by_stock.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # RMSE时间趋势
    plt.figure(figsize=(12, 6))
    monthly_metrics['RMSE'].plot()
    plt.title('预测RMSE时间趋势', fontsize=15)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.tight_layout()
    plt.savefig('result/rmse_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 收益区间分析
    bins = [-np.inf, -0.05, 0, 0.05, np.inf]
    labels = ['<-5%', '-5%-0%', '0-5%', '>5%']
    df['return_bin'] = pd.cut(df['actual_return'], bins=bins, labels=labels)
    bin_metrics = df.groupby('return_bin').apply(
        lambda x: pd.Series(calculate_metrics(x['actual_return'], x['predicted_return'])))
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(bin_metrics[['MAE', 'RMSE', 'Direction Accuracy (%)']], 
                annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title('不同收益区间的预测准确性', fontsize=15)
    plt.xlabel('指标', fontsize=12)
    plt.ylabel('收益区间', fontsize=12)
    plt.tight_layout()
    plt.savefig('result/accuracy_by_return_bin.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. 主执行流程
if __name__ == "__main__":
    # 加载数据
    actual_df, predicted_df = load_data()
    
    # 合并数据
    print("\n正在合并数据...")
    merged_df = pd.merge(actual_df, predicted_df, on=['datetime', 'instrument'], how='inner')
    merged_df = merged_df.rename(columns={'LABEL0': 'actual_return', 'score': 'predicted_return'})
    
    # 数据清洗
    initial_count = len(merged_df)
    merged_df = merged_df.dropna(subset=['actual_return', 'predicted_return'])
    print(f"\n已删除 {initial_count - len(merged_df)} 条包含缺失值的记录")
    
    # 计算误差
    merged_df['error'] = merged_df['actual_return'] - merged_df['predicted_return']
    
    # 计算整体指标
    print("\n计算整体指标...")
    overall_metrics = calculate_metrics(merged_df['actual_return'], merged_df['predicted_return'])
    print(pd.DataFrame.from_dict(overall_metrics, orient='index', columns=['值']))
    
    # 计算按股票指标
    print("\n计算按股票指标...")
    stock_metrics = merged_df.groupby('instrument').apply(
        lambda x: pd.Series(calculate_metrics(x['actual_return'], x['predicted_return'])))
    
    # 计算按月指标
    print("\n计算按月指标...")
    merged_df['month'] = merged_df['datetime'].dt.to_period('M')
    monthly_metrics = merged_df.groupby('month').apply(
        lambda x: pd.Series(calculate_metrics(x['actual_return'], x['predicted_return'])))
    
    # 生成可视化
    print("\n生成可视化图表...")
    create_visualizations(merged_df, stock_metrics, monthly_metrics)
    
    # 保存结果
    print("\n保存分析结果...")
    merged_df.to_csv('result/merged_returns_with_errors.csv', index=False, encoding='utf-8-sig')
    stock_metrics.to_csv('result/stock_wise_accuracy.csv', encoding='utf-8-sig')
    monthly_metrics.to_csv('result/monthly_accuracy.csv', encoding='utf-8-sig')
    
    print("\n分析完成！所有结果已保存到当前目录。")
