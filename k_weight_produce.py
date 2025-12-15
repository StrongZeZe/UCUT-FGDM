import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import os
import pickle


def configure_sci_style():
    """Configure matplotlib for SCI paper style"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (7, 5),
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8
    })


def run_expert_simulation(position, excel_filename, chart_filename, random_seed):
    """
    Main function to run expert assessment simulation

    Parameters:
    position (str): Directory path for saving results
    excel_filename (str): Excel filename
    chart_filename (str): Chart filename
    random_seed (int): Random seed for reproducibility

    Returns:
    tuple: (k_values, assessment_array)
        k_values: 1D array of K values for each decision maker
        assessment_array: 3D array of shape (num_simulations, num_experts, num_weights)
    """
    configure_sci_style()

    # Define file paths
    excel_path = os.path.join(position, excel_filename)
    chart_path = os.path.join(position, chart_filename)

    # Ensure output directory exists
    if not os.path.exists(position):
        os.makedirs(position)
        print(f"Created directory: {position}")

    # True weights
    w_true = np.array([0.139, 0.267, 0.216, 0.133, 0.107, 0.138])

    # Randomly generate K values for 5 decision makers
    np.random.seed(random_seed)
    k_values = np.random.randint(50, 251, 5)

    print("Generated K values for 5 decision makers:")
    for i, k in enumerate(k_values):
        print(f"DM {i + 1}: K = {k}")

    # Simulation parameters
    num_simulations = 500
    num_experts = len(k_values)
    num_weights = len(w_true)

    # Initialize 3D array for all assessments
    assessment_array = np.zeros((num_simulations, num_experts, num_weights))

    print(f"\nRunning {num_simulations} simulations...")

    # Run simulations
    for sim in range(num_simulations):
        for i, k in enumerate(k_values):
            alpha = k * w_true
            assessed_weights = np.random.dirichlet(alpha)
            assessment_array[sim, i] = assessed_weights

        # Display first 3 simulations
        if sim < 3:
            print(f"\n--- Simulation {sim + 1} ---")
            for i, k in enumerate(k_values):
                weights = assessment_array[sim, i]
                js_dist = jensenshannon(w_true, weights)
                print(f"DM {i + 1} (K={k}): {weights.round(4)} | JS: {js_dist:.4f}")

    print(f"\nCompleted {num_simulations} simulations")

    # Analyze and save results
    analysis_results = analyze_simulation_results(w_true, k_values, assessment_array, excel_path, chart_path)

    return k_values, assessment_array


def analyze_simulation_results(w_true, k_values, assessment_array, excel_path, chart_path):
    """
    Analyze simulation results and generate outputs
    """
    num_simulations, num_experts, num_weights = assessment_array.shape

    # Calculate JS distances for all simulations
    js_distances = np.zeros((num_simulations, num_experts))
    for sim in range(num_simulations):
        for i in range(num_experts):
            js_distances[sim, i] = jensenshannon(w_true, assessment_array[sim, i])

    # Statistical analysis
    stats_data = []
    for i in range(num_experts):
        dm_js = js_distances[:, i]
        stats = {
            'Decision_Maker': f'DM{i + 1}',
            'K_Value': k_values[i],
            'Mean_JS_Distance': np.mean(dm_js),
            'Std_JS_Distance': np.std(dm_js),
            'Min_JS_Distance': np.min(dm_js),
            'Max_JS_Distance': np.max(dm_js),
            'Median_JS_Distance': np.median(dm_js),
            'Q1_JS_Distance': np.percentile(dm_js, 25),
            'Q3_JS_Distance': np.percentile(dm_js, 75)
        }
        stats_data.append(stats)

    # Create comprehensive visualization
    create_sci_visualization(w_true, k_values, assessment_array, js_distances, chart_path)

    # Save to Excel
    save_to_excel(w_true, k_values, assessment_array, js_distances, stats_data, excel_path)

    print(f"Results saved to: {excel_path}")
    print(f"Chart saved to: {chart_path}")

    return stats_data


def create_sci_visualization(w_true, k_values, assessment_array, js_distances, chart_path):
    """
    Create SCI-style visualization
    """
    num_simulations, num_experts, num_weights = assessment_array.shape

    # Create figure with subplots
    fig = plt.figure(figsize=(12, 10))

    # Color scheme for SCI papers (grayscale-friendly)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    marker_styles = ['o', 's', '^', 'D', 'v']

    # Plot 1: JS Distance vs K Value
    ax1 = plt.subplot(2, 2, 1)
    mean_js = np.mean(js_distances, axis=0)

    for i in range(num_experts):
        ax1.scatter(k_values[i], mean_js[i],
                    color=colors[i], marker=marker_styles[i],
                    s=80, alpha=0.8, label=f'DM{i + 1}')

    # Add trend line
    k_fit = np.linspace(min(k_values), max(k_values), 100)
    js_fit = 0.5 / np.sqrt(k_fit)  # Theoretical relationship
    ax1.plot(k_fit, js_fit, 'k--', alpha=0.7, linewidth=1, label='Theoretical trend')

    ax1.set_xlabel('K Value (Expertise Level)')
    ax1.set_ylabel('Mean JS Distance')
    ax1.set_title('(a) Expertise vs Assessment Accuracy')
    ax1.legend(frameon=True, fancybox=False, edgecolor='black')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Weight distributions (last simulation)
    ax2 = plt.subplot(2, 2, 2)
    x_pos = np.arange(num_weights)
    width = 0.15

    for i in range(num_experts):
        ax2.bar(x_pos + i * width, assessment_array[-1, i], width,
                color=colors[i], alpha=0.7, label=f'DM{i + 1} (K={k_values[i]})')

    # True weights
    ax2.plot(x_pos + (num_experts - 1) * width / 2, w_true, 'ko-',
             linewidth=2, markersize=6, label='True Weights')

    ax2.set_xlabel('Alternatives')
    ax2.set_ylabel('Weight Values')
    ax2.set_title('(b) Final Simulation Weights')
    ax2.set_xticks(x_pos + (num_experts - 1) * width / 2)
    ax2.set_xticklabels([f'A{i + 1}' for i in range(num_weights)])
    ax2.legend(frameon=True, fancybox=False, edgecolor='black')
    ax2.grid(True, alpha=0.3)

    # Plot 3: JS Distance distribution
    ax3 = plt.subplot(2, 2, 3)
    box_data = [js_distances[:, i] for i in range(num_experts)]
    box_labels = [f'DM{i + 1}\n(K={k})' for i, k in enumerate(k_values)]

    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)

    # Customize boxplot colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for element in ['whiskers', 'caps', 'medians']:
        for line in bp[element]:
            line.set_color('black')
            line.set_linewidth(1)

    ax3.set_ylabel('JS Distance')
    ax3.set_title('(c) JS Distance Distribution')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Convergence analysis
    ax4 = plt.subplot(2, 2, 4)

    # Calculate cumulative mean JS distance
    cumulative_means = np.zeros((num_simulations, num_experts))
    for i in range(num_experts):
        cumulative_means[:, i] = np.cumsum(js_distances[:, i]) / np.arange(1, num_simulations + 1)

    for i in range(num_experts):
        ax4.plot(range(1, num_simulations + 1), cumulative_means[:, i],
                 color=colors[i], linewidth=1.5, label=f'DM{i + 1}')

    ax4.set_xlabel('Number of Simulations')
    ax4.set_ylabel('Cumulative Mean JS Distance')
    ax4.set_title('(d) Convergence Analysis')
    ax4.legend(frameon=True, fancybox=False, edgecolor='black')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')  # Log scale for better visualization

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    #plt.show()


def save_to_excel(w_true, k_values, assessment_array, js_distances, stats_data, excel_path):
    """
    Save all results to Excel file
    """
    num_simulations, num_experts, num_weights = assessment_array.shape

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # Sheet 1: K values information
        k_df = pd.DataFrame({
            'Decision_Maker': [f'DM{i + 1}' for i in range(num_experts)],
            'K_Value': k_values,
            'Description': ['Higher K indicates better expertise'] * num_experts
        })
        k_df.to_excel(writer, sheet_name='K_Values', index=False)

        # Sheet 2: True weights
        true_df = pd.DataFrame({
            'Alternative': [f'A{i + 1}' for i in range(num_weights)],
            'True_Weight': w_true
        })
        true_df.to_excel(writer, sheet_name='True_Weights', index=False)

        # Sheet 3: All simulation results
        sim_data = []
        for sim in range(num_simulations):
            for i in range(num_experts):
                row = {
                    'Simulation': sim + 1,
                    'Decision_Maker': f'DM{i + 1}',
                    'K_Value': k_values[i],
                    'JS_Distance': js_distances[sim, i]
                }
                # Add weight for each alternative
                for j in range(num_weights):
                    row[f'Weight_A{j + 1}'] = assessment_array[sim, i, j]
                sim_data.append(row)

        sim_df = pd.DataFrame(sim_data)
        sim_df.to_excel(writer, sheet_name='All_Simulations', index=False)

        # Sheet 4: Statistical summary
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        # Sheet 5: First 50 simulations (for quick inspection)
        first_50 = sim_df[sim_df['Simulation'] <= 50]
        first_50.to_excel(writer, sheet_name='First_50_Simulations', index=False)


def save_table_values(data, file_path):
    """将数据保存为Python数据文件"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to: {file_path}")


def load_table_values(file_path):
    """从文件加载数据"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def example_usage():
    """
    Example of how to use the simulation function
    """
    # Define your storage position
    position = r"E:\newManucript\python_code_rare\script2\data\simulation_data"
    excel_filenames = ["simulation_results1.xlsx", "simulation_results2.xlsx",
                       "simulation_results3.xlsx", "simulation_results4.xlsx",
                       "simulation_results5.xlsx"]
    chart_filenames = ["simulation_analysis1.png", "simulation_analysis2.png",
                       "simulation_analysis3.png", "simulation_analysis4.png",
                       "simulation_analysis5.png"]
    random_list = [41, 42, 43, 44, 45]

    k_values_list = []
    assessment_array_list = []

    # Ensure the directory exists
    if not os.path.exists(position):
        os.makedirs(position)

    # Run simulation 5 times
    for i in range(5):
        print(f"\n=== Running simulation batch {i + 1} ===")

        k_values, assessment_array = run_expert_simulation(
            position, excel_filenames[i], chart_filenames[i], random_list[i]
        )

        # Save assessment arrays and k values with proper file paths
        assessment_file_path = os.path.join(position, f"assessment_array_{i + 1}.pkl")
        k_value_file_path = os.path.join(position, f"k_values_{i + 1}.pkl")

        save_table_values(assessment_array, assessment_file_path)
        save_table_values(k_values, k_value_file_path)

        k_values_list.append(k_values)
        assessment_array_list.append(assessment_array)

        print(f"Completed batch {i + 1}/5")

    return k_values_list, assessment_array_list


if __name__ == "__main__":
    # Run example
    k_vals, assessments = example_usage()
    print(f"\nTotal batches completed: {len(assessments)}")
    print(f"Shape of first assessment array: {assessments[0].shape}")