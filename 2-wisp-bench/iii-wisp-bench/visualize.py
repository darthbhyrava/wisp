#!/usr/bin/env python3
"""
WISP-Bench Results Visualization Script
Generates comprehensive visualizations from WISP-bench scoring results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class WISPBenchVisualizer:
    def __init__(self, results_file: str):
        """Initialize with results JSON file."""
        with open(results_file, 'r') as f:
            self.results = json.load(f)

        # set up plotting style to match paper aesthetic
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 11,
            'font.family': 'sans-serif',
            'axes.linewidth': 0.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linewidth': 0.5,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white'
        })
        self.fig_size = (12, 8)

        # extract key data structures
        self.methods = list(self.results['method_scores'].keys())

        # assign a diverse, colorblind-safe palette to methods
        base_palette = sns.color_palette('tab10', n_colors=max(10, len(self.methods)))
        if len(self.methods) > 10:
            base_palette = sns.color_palette('tab20', n_colors=len(self.methods))
        self.method_colors = {m: base_palette[i] for i, m in enumerate(self.methods)}

        # series colors for multi-series plots (keep one blue, use complementary colors for the rest)
        self.series_colors = {
            'macro': '#1f77b4',      # blue
            'weighted': '#ff7f0e',   # orange
            'composite': '#2ca02c',  # green
            'pure': '#9467bd',       # purple
        }

    def _display_method(self, name: str) -> str:
        """Map internal method names to display labels."""
        return 'wisp-ify' if name == 'melanie' else name

    def plot_1_method_performance_comparison(self):
        """Method performance showing impact of scoring adjustments."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        # extract data
        methods = []
        macro_scores = []
        weighted_scores = []
        composite_scores = []
        pure_scores = []

        for method, data in self.results['method_scores'].items():
            methods.append(method)
            macro_scores.append(data['macro_score'])
            weighted_scores.append(data['weighted_macro_score'])
            composite_scores.append(data['composite_score'])
            pure_scores.append(data['non_ocr_macro_score'])

        # grouped bars with diverse series colors
        x = np.arange(len(methods))
        width = 0.2

        bars1 = ax.bar(x - 1.5 * width, macro_scores, width, label='Macro Score',
                       color=self.series_colors['macro'], alpha=0.9)
        bars2 = ax.bar(x - 0.5 * width, weighted_scores, width, label='Weighted Score',
                       color=self.series_colors['weighted'], alpha=0.9)
        bars3 = ax.bar(x + 0.5 * width, composite_scores, width, label='Composite Score',
                       color=self.series_colors['composite'], alpha=0.9)
        bars4 = ax.bar(x + 1.5 * width, pure_scores, width, label='Pure (Non-OCR) Score',
                       color=self.series_colors['pure'], alpha=0.9)

        ax.set_xlabel('Methods')
        ax.set_ylabel('Score (%)')
        ax.set_title('Method Performance: Impact of Scoring Adjustments', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([self._display_method(m) for m in methods], rotation=45, ha='right')
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

        # labels
        for bars in [bars1, bars2, bars3, bars4]:
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', 
                            fontsize=8, rotation=90)

        plt.tight_layout()
        return fig

    def plot_2_wisp_category_performance(self):
        """WISP label performance by method (grouped bar chart)."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        # extract WISP category data (exclude OCR_ERROR)
        wisp_categories = [cat for cat in self.results['wisp_label_scores'].keys()
                           if cat != 'OCR_ERROR']

        # create data matrix
        data_matrix = []
        for category in wisp_categories:
            row = []
            for method in self.methods:
                if method in self.results['wisp_label_scores'][category]:
                    score = self.results['wisp_label_scores'][category][method]['macro_score']
                    row.append(score)
                else:
                    row.append(0)
            data_matrix.append(row)

        x = np.arange(len(wisp_categories))
        width = 0.12

        for i, method in enumerate(self.methods):
            scores = [data_matrix[j][i] for j in range(len(wisp_categories))]
            bars = ax.bar(x + i * width - (len(self.methods) - 1) * width / 2, scores,
                          width, label=self._display_method(method), alpha=0.9,
                          color=self.method_colors[method])

            for bar in bars:
                h = bar.get_height()
                if h > 5:
                    ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3), textcoords='offset points',
                                ha='center', va='bottom', fontsize=8, rotation=90)

        ax.set_xlabel('WISP Categories')
        ax.set_ylabel('Macro Score (%)')
        ax.set_title('WISP Category Performance by Method', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(wisp_categories)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def plot_2b_method_scoring_comparison(self):
        """WISP category performance by method with scoring type variations shown as shading."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        # extract WISP category data (exclude OCR_ERROR)
        wisp_categories = [cat for cat in self.results['wisp_label_scores'].keys()
                           if cat != 'OCR_ERROR']

        # for WISP categories, we'll use the macro scores but show different shadings
        # to represent the overall method performance variations
        x = np.arange(len(wisp_categories))
        method_group_width = 0.8  # total width for all methods in a category
        method_separation = 0.05  # space between method groups
        n_methods = len(self.methods)
        method_width = (method_group_width - method_separation * (n_methods - 1)) / n_methods
        scoring_width = method_width / 4

        # more intuitive shading patterns
        scoring_patterns = {
            'macro': {'alpha': 1.0, 'hatch': None},           # Solid (strongest)
            'weighted': {'alpha': 0.85, 'hatch': '///'},      # Diagonal lines
            'composite': {'alpha': 0.7, 'hatch': '+++'},      # Plus signs
            'pure': {'alpha': 0.55, 'hatch': '...'}           # Dots (lightest)
        }

        # plot bars for each method with different shadings representing scoring variations
        method_legend_handles = []
        scoring_legend_handles = []
        
        for i, method in enumerate(self.methods):
            base_color = self.method_colors[method]
            method_start = x + i * (method_width + method_separation) - method_group_width/2
            
            # get the macro scores for this method across WISP categories
            scores = []
            for category in wisp_categories:
                if method in self.results['wisp_label_scores'][category]:
                    score = self.results['wisp_label_scores'][category][method]['macro_score']
                    scores.append(score)
                else:
                    scores.append(0)
            
            # get the method's overall performance variations for shading reference
            method_data = self.results['method_scores'][method]
            macro_base = method_data['macro_score']
            weighted_ratio = method_data['weighted_macro_score'] / macro_base if macro_base > 0 else 1
            composite_ratio = method_data['composite_score'] / macro_base if macro_base > 0 else 1
            pure_ratio = method_data['non_ocr_macro_score'] / macro_base if macro_base > 0 else 1
            
            # create 4 sub-bars with different shading patterns
            # macro (solid)
            bars1 = ax.bar(method_start, scores, scoring_width, 
                          color=base_color, **scoring_patterns['macro'])
            
            # weighted (diagonal lines)
            weighted_scores = [s * weighted_ratio for s in scores]
            bars2 = ax.bar(method_start + scoring_width, weighted_scores, scoring_width,
                          color=base_color, **scoring_patterns['weighted'])
            
            # composite (plus signs)
            composite_scores = [s * composite_ratio for s in scores]
            bars3 = ax.bar(method_start + 2*scoring_width, composite_scores, scoring_width,
                          color=base_color, **scoring_patterns['composite'])
            
            # pure (dots)
            pure_scores = [s * pure_ratio for s in scores]
            bars4 = ax.bar(method_start + 3*scoring_width, pure_scores, scoring_width,
                          color=base_color, **scoring_patterns['pure'])

            # add to legend (only once per method)
            if i == 0:
                method_legend_handles.append(bars1[0])  # Use first bar for method color
                scoring_legend_handles.extend([bars1[0], bars2[0], bars3[0], bars4[0]])

        # create method legend handles for all methods
        from matplotlib.patches import Patch
        method_patches = [Patch(color=self.method_colors[method], alpha=0.8) 
                         for method in self.methods]
        method_labels = [self._display_method(m) for m in self.methods]
        
        # create scoring pattern legend handles
        scoring_patches = []
        for pattern_name, pattern_props in scoring_patterns.items():
            patch = Patch(color='gray', **pattern_props)
            scoring_patches.append(patch)
        scoring_labels = ['Macro', 'Weighted', 'Composite', 'Pure']

        ax.set_xlabel('WISP Categories')
        ax.set_ylabel('Score (%)')
        ax.set_title('WISP Category Performance: Methods with Scoring Variations', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(wisp_categories)
        
        # create two legends with even smaller font size and better positioning
        leg1 = ax.legend(method_patches, method_labels, title="Methods", 
                        loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=6, title_fontsize=7,
                        frameon=False, borderaxespad=0, handlelength=0.8, handletextpad=0.3)
        leg2 = ax.legend(scoring_patches, scoring_labels, title="Scoring Types", 
                        loc='upper left', bbox_to_anchor=(1.02, 0.5), fontsize=6, title_fontsize=7,
                        frameon=False, borderaxespad=0, handlelength=0.8, handletextpad=0.3)
        ax.add_artist(leg1)  # Add back the first legend

        plt.tight_layout()
        # adjust layout to ensure legends fit with more space
        plt.subplots_adjust(right=0.75)
        return fig

    def plot_3_method_performance_heatmap(self):
        """Method Performance Heatmap: options x methods."""
        fig, ax = plt.subplots(figsize=(14, 10))

        # order options logically (presence -> fuzzy -> exact)
        option_order = []
        prefixes = ['line_breaks', 'vertical', 'prefix', 'internal']
        suffixes = ['_presence', '_fuzzy', '_exact']

        for prefix in prefixes:
            for suffix in suffixes:
                option_name = prefix + suffix
                if any(option_name in method_data['option_scores']
                       for method_data in self.results['method_scores'].values()):
                    option_order.append(option_name)

        option_order.append('ocr_error')

        # create data matrix
        data_matrix = []
        available_options = []

        for option in option_order:
            row = []
            option_exists = False
            for method in self.methods:
                if option in self.results['method_scores'][method]['option_scores']:
                    score = self.results['method_scores'][method]['option_scores'][option]['pass_rate']
                    row.append(score)
                    option_exists = True
                else:
                    row.append(np.nan)

            if option_exists:
                data_matrix.append(row)
                available_options.append(option)

        data_df = pd.DataFrame(data_matrix, index=available_options, columns=self.methods)

        # single color (blue) with different shading levels - mild to strong
        from matplotlib.colors import LinearSegmentedColormap
        colors_blue_shades = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c']
        blue_shades_cmap = LinearSegmentedColormap.from_list('blue_shades', colors_blue_shades)

        sns.heatmap(data_df, annot=True, fmt='.0f', cmap=blue_shades_cmap,
                    center=50, ax=ax, cbar_kws={'label': 'Pass Rate (%)'})

        ax.set_title('Method Performance Heatmap: Pass Rates by Option', pad=20)
        ax.set_xlabel('Methods')
        ax.set_ylabel('Options')
        ax.set_xticklabels([self._display_method(m) for m in self.methods], rotation=45, ha='right')

        plt.tight_layout()
        return fig

    def plot_4_performance_vs_reliability_scatter(self):
        """Performance vs Reliability Scatter Plot."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        macro_scores = []
        composite_scores = []
        annotation_counts = []
        method_names = []

        for method, data in self.results['method_scores'].items():
            macro_scores.append(data['macro_score'])
            composite_scores.append(data['composite_score'])
            annotation_counts.append(data['total_annotations'])
            method_names.append(method)

        ax.scatter(macro_scores, composite_scores,
                   s=[count * 3 for count in annotation_counts],
                   c=[self.method_colors[method] for method in method_names],
                   alpha=0.7, edgecolors='black', linewidth=1)

        for i, method in enumerate(method_names):
            ax.annotate(self._display_method(method), (macro_scores[i], composite_scores[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9, ha='left')

        min_val = min(min(macro_scores), min(composite_scores))
        max_val = max(max(macro_scores), max(composite_scores))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal Performance')

        ax.set_xlabel('Macro Score (%)')
        ax.set_ylabel('Composite Score (Reliability-Adjusted) (%)')
        ax.set_title('Performance vs Reliability')
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def plot_5_poem_winner_pie_chart(self):
        """Poem 'winner' distribution as horizontal bar chart with counts."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        wins_data = self.results['winner_analysis']['method_wins']
        # sort methods by number of wins (descending)
        sorted_items = sorted(wins_data.items(), key=lambda x: x[1], reverse=True)
        methods = [item[0] for item in sorted_items]
        wins = [item[1] for item in sorted_items]
        colors = [self.method_colors[method] for method in methods]
        display_methods = [self._display_method(m) for m in methods]

        # create horizontal bars
        y_pos = np.arange(len(methods))
        bars = ax.barh(y_pos, wins, color=colors, alpha=0.8)
        
        # add count labels on bars
        for i, (bar, count) in enumerate(zip(bars, wins)):
            width = bar.get_width()
            ax.annotate(f'{int(count)}',
                       xy=(width, bar.get_y() + bar.get_height() / 2),
                       xytext=(3, 0), textcoords='offset points',
                       ha='left', va='center', fontsize=10)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_methods)
        ax.set_xlabel('Number of Poem Wins')
        ax.set_title('Poem Winner Distribution\n(Number of poems where method had highest score)')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def plot_6_option_winner_horizontal_bars(self):
        """Option-Level Winner Horizontal Bar Chart."""
        fig, ax = plt.subplots(figsize=(12, 10))

        option_wins = {}
        for method, options_dict in self.results['winner_analysis']['method_option_wins'].items():
            for option, wins in options_dict.items():
                if not option.startswith('ocr_error'):
                    if option not in option_wins:
                        option_wins[option] = {}
                    option_wins[option][method] = wins

        option_order = []
        prefixes = ['line_breaks', 'vertical', 'prefix', 'internal']
        suffixes = ['_presence', '_fuzzy', '_exact']
        for prefix in prefixes:
            for suffix in suffixes:
                option_name = prefix + suffix
                if option_name in option_wins:
                    option_order.append(option_name)

        y_pos = np.arange(len(option_order))
        bottom_values = np.zeros(len(option_order))

        for method in self.methods:
            values = [option_wins.get(option, {}).get(method, 0) for option in option_order]
            ax.barh(y_pos, values, left=bottom_values,
                    label=self._display_method(method), color=self.method_colors[method], alpha=0.8)
            bottom_values += values

        ax.set_yticks(y_pos)
        ax.set_yticklabels(option_order)
        ax.set_xlabel('Total Wins')
        ax.set_title('Option-Level Winner Analysis')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def plot_7_wisp_category_winner_bars(self):
        """WISP Category Winner Horizontal Bar Chart."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        wisp_wins = self.results['winner_analysis']['method_wisp_wins']
        # extract WISP category data (exclude OCR_ERROR) - same as other plots
        wisp_categories = [cat for cat in self.results['wisp_label_scores'].keys()
                          if cat != 'OCR_ERROR']

        y_pos = np.arange(len(wisp_categories))
        bottom_values = np.zeros(len(wisp_categories))

        for method in self.methods:
            values = [wisp_wins.get(method, {}).get(category, 0) for category in wisp_categories]
            ax.barh(y_pos, values, left=bottom_values,
                    label=self._display_method(method), color=self.method_colors[method], alpha=0.8)
            bottom_values += values

        ax.set_yticks(y_pos)
        ax.set_yticklabels(wisp_categories)
        ax.set_xlabel('Total Wins')
        ax.set_title('WISP Category Winner Analysis')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        return fig

    def plot_8_radar_chart_wisp_performance(self):
        """Radar Chart for Method Performance across WISP Categories."""
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # extract WISP category data (exclude OCR_ERROR) - same as other plots
        categories = [cat for cat in self.results['wisp_label_scores'].keys()
                     if cat != 'OCR_ERROR']
        
        # if no categories available, return empty plot
        if not categories:
            ax.text(0.5, 0.5, 'No WISP categories available', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=16)
            return fig
            
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        for method in self.methods:
            values = []
            for category in categories:
                if category in self.results['wisp_label_scores'] and method in self.results['wisp_label_scores'][category]:
                    score = self.results['wisp_label_scores'][category][method]['macro_score']
                    values.append(score)
                else:
                    values.append(0)
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2,
                    label=self._display_method(method), color=self.method_colors[method])
            ax.fill(angles, values, alpha=0.25, color=self.method_colors[method])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)

        ax.set_title('Method Performance Across WISP Categories', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()
        return fig

    def plot_9_ocr_penalty_analysis(self):
        """OCR Penalty Analysis: Catastrophic/Mixed/Pure rates."""
        fig, ax = plt.subplots(figsize=self.fig_size)

        methods = []
        catastrophic_rates = []
        mixed_rates = []
        pure_rates = []

        for method, data in self.results['method_scores'].items():
            methods.append(method)
            penalty_data = data['ocr_penalty']
            catastrophic_rates.append(penalty_data['catastrophic_rate'] * 100)
            mixed_rates.append(penalty_data['mixed_rate'] * 100)
            pure_rates.append(penalty_data['pure_rate'] * 100)

        x = np.arange(len(methods))
        width = 0.6

        ax.bar(x, catastrophic_rates, width, label='Catastrophic', color='#D73027', alpha=0.9)
        ax.bar(x, mixed_rates, width, bottom=catastrophic_rates, label='Mixed', color='#FEE08B', alpha=0.9)
        ax.bar(x, pure_rates, width, bottom=np.array(catastrophic_rates) + np.array(mixed_rates),
               label='Pure', color=self.series_colors['composite'], alpha=0.9)

        for i, (cat, mix, pure) in enumerate(zip(catastrophic_rates, mixed_rates, pure_rates)):
            if cat > 2:
                ax.text(i, cat / 2, f'{cat:.0f}%', ha='center', va='center', fontweight='bold', fontsize=9)
            if mix > 2:
                ax.text(i, cat + mix / 2, f'{mix:.0f}%', ha='center', va='center', fontweight='bold', fontsize=9)
            if pure > 10:
                ax.text(i, cat + mix + pure / 2, f'{pure:.0f}%', ha='center', va='center', fontweight='bold', fontsize=9)

        ax.set_xlabel('Methods')
        ax.set_ylabel('Percentage of Annotations')
        ax.set_title('OCR Penalty Analysis: Annotation Quality Distribution', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([self._display_method(m) for m in methods], rotation=45, ha='right')
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        return fig

    def generate_all_plots(self, output_dir='plots'):
        """Generate all plots and save them."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        plots = [
            (self.plot_1_method_performance_comparison, '1_method_performance_comparison'),
            (self.plot_2_wisp_category_performance, '2_wisp_category_performance'),
            (self.plot_2b_method_scoring_comparison, '2b_method_scoring_comparison'),
            (self.plot_3_method_performance_heatmap, '3_method_performance_heatmap'),
            (self.plot_4_performance_vs_reliability_scatter, '4_performance_vs_reliability_scatter'),
            (self.plot_5_poem_winner_pie_chart, '5_poem_winner_bar_chart'),
            (self.plot_6_option_winner_horizontal_bars, '6_option_winner_horizontal_bars'),
            (self.plot_7_wisp_category_winner_bars, '7_wisp_category_winner_bars'),
            (self.plot_8_radar_chart_wisp_performance, '8_radar_chart_wisp_performance'),
            (self.plot_9_ocr_penalty_analysis, '9_ocr_penalty_analysis'),
        ]

        for plot_func, filename in plots:
            print(f"Generating {filename}...")
            fig = plot_func()
            fig.savefig(f'{output_dir}/{filename}.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

        print(f"All plots saved to {output_dir}/")

    def show_plots_interactive(self):
        """Show all plots in interactive mode."""
        plots = [
            (self.plot_1_method_performance_comparison, 'Method Performance Comparison'),
            (self.plot_2_wisp_category_performance, 'WISP Category Performance'),
            (self.plot_2b_method_scoring_comparison, 'Method Scoring Type Comparison'),
            (self.plot_3_method_performance_heatmap, 'Method Performance Heatmap'),
            (self.plot_4_performance_vs_reliability_scatter, 'Performance vs Reliability'),
            (self.plot_5_poem_winner_pie_chart, 'Poem Winner Distribution (Bar Chart)'),
            (self.plot_6_option_winner_horizontal_bars, 'Option-Level Winners'),
            (self.plot_7_wisp_category_winner_bars, 'WISP Category Winners'),
            (self.plot_8_radar_chart_wisp_performance, 'Radar Chart WISP Performance'),
            (self.plot_9_ocr_penalty_analysis, 'OCR Penalty Analysis'),
        ]

        for plot_func, title in plots:
            print(f"Showing: {title}")
            fig = plot_func()
            plt.show()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='WISP-Bench Results Visualizer')
    parser.add_argument('results_file', help='Path to JSON results file')
    parser.add_argument('--output-dir', default='plots', help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively instead of saving')
    parser.add_argument('--plot', type=int, choices=list(range(1, 10)) + [21],
                       help='Generate specific plot only (1-9, 21 for method scoring comparison)')

    args = parser.parse_args()

    viz = WISPBenchVisualizer(args.results_file)

    if args.plot:
        plot_methods = [
            viz.plot_1_method_performance_comparison,
            viz.plot_2_wisp_category_performance,
            viz.plot_3_method_performance_heatmap,
            viz.plot_4_performance_vs_reliability_scatter,
            viz.plot_5_poem_winner_pie_chart,
            viz.plot_6_option_winner_horizontal_bars,
            viz.plot_7_wisp_category_winner_bars,
            viz.plot_8_radar_chart_wisp_performance,
            viz.plot_9_ocr_penalty_analysis,
        ]
        
        # special handling for plot 21 (2b)
        if args.plot == 21:
            fig = viz.plot_2b_method_scoring_comparison()
        else:
            fig = plot_methods[args.plot - 1]()
        if args.show:
            plt.show()
        else:
            import os
            os.makedirs(args.output_dir, exist_ok=True)
            if args.plot == 21:
                filename = f'{args.output_dir}/2b_method_scoring_comparison.png'
            else:
                filename = f'{args.output_dir}/plot_{args.plot}.png'
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {filename}")

    elif args.show:
        viz.show_plots_interactive()
    else:
        viz.generate_all_plots(args.output_dir)


if __name__ == "__main__":
    main()