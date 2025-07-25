import pandas as pd
import matplotlib.pyplot as plt

def plot_multiple_loads(csv_files, labels):
    """
    绘制多个负载日志文件的对比图
    
    Args:
        csv_files: CSV文件路径列表
        labels: 对应的标签列表
    """
    plt.figure(figsize=(12, 8))
    
    # 设置线条样式
    styles = ['-', '--']  # 实线和虚线
    colors = {
        'cpu': 'red',     # CPU负载用红色
        'ram': 'blue',    # RAM负载用蓝色
        'bw': 'green'     # 带宽负载用绿色
    }
    
    # 读取并绘制每个文件的数据
    for i, (file, label) in enumerate(zip(csv_files, labels)):
        df = pd.read_csv(file)
        
        # 绘制 CPU 负载
        plt.plot(df['nodes'], df['cpu_load'], 
                linestyle=styles[i], color=colors['cpu'], 
                label=f'{label} - CPU')
        
        # 绘制 RAM 负载
        plt.plot(df['nodes'], df['ram_load'], 
                linestyle=styles[i], color=colors['ram'], 
                label=f'{label} - RAM')
        
        # 绘制带宽负载
        plt.plot(df['nodes'], df['bw_load'], 
                linestyle=styles[i], color=colors['bw'], 
                label=f'{label} - BW')
    
    # 设置图表属性
    plt.xlabel('number of nodes')
    plt.ylabel('ratio of load')
    plt.title('resource load comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局以显示完整的图例
    plt.tight_layout()
    
    # 保存图表
    plt.savefig('load_comparison.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 load_comparison.png")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    # 文件路径
    files = [
        'load_log_20250611_191156.csv',
        'load_log_20250611_192403.csv'
    ]
    
    # 为每个文件添加标签
    labels = ['random', 'RA']
    
    # 绘制图表
    plot_multiple_loads(files, labels)