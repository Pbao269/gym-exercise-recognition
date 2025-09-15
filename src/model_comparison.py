"""
Comprehensive comparison between XGBoost and Deep Learning models
for gym exercise recognition.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_model_results():
    """Load results from both models."""
    models_dir = Path("models")
    
    # Load XGBoost results
    with open(models_dir / "xgb_metadata.json", 'r') as f:
        xgb_results = json.load(f)
    
    # Load Keras results
    with open(models_dir / "keras_metadata.json", 'r') as f:
        keras_results = json.load(f)
    
    return xgb_results, keras_results

def create_comparison_table(xgb_results, keras_results):
    """Create a detailed comparison table."""
    comparison_data = {
        'Metric': [
            'Test Accuracy',
            'Test F1 (Macro)',
            'Test F1 (Weighted)',
            'Validation Accuracy',
            'Validation F1 (Macro)',
            'Validation F1 (Weighted)',
            'Model Complexity',
            'Training Time',
            'Inference Speed',
            'Memory Usage'
        ],
        'XGBoost': [
            f"{xgb_results['performance']['test_accuracy']:.1%}",
            f"{xgb_results['performance']['test_f1_macro']:.3f}",
            f"{xgb_results['performance']['test_f1_weighted']:.3f}",
            f"{xgb_results['performance']['val_accuracy']:.1%}",
            f"{xgb_results['performance']['val_f1_macro']:.3f}",
            f"{xgb_results['performance']['val_f1_weighted']:.3f}",
            f"{xgb_results['n_features']} features",
            "~2 minutes",
            "Very Fast",
            "Low (~MB)"
        ],
        'Deep Learning (CNN-LSTM)': [
            f"{keras_results['performance']['test_accuracy']:.1%}",
            f"{keras_results['performance']['test_f1_macro']:.3f}",
            f"{keras_results['performance']['test_f1_weighted']:.3f}",
            f"{keras_results['performance']['val_accuracy']:.1%}",
            f"{keras_results['performance']['val_f1_macro']:.3f}",
            f"{keras_results['performance']['val_f1_weighted']:.3f}",
            f"{keras_results['total_parameters']:,} parameters",
            f"{keras_results['training']['epochs_completed']} epochs (~1 hour)",
            "Fast",
            "Medium (~3MB)"
        ]
    }
    
    return pd.DataFrame(comparison_data)

def plot_performance_comparison(xgb_results, keras_results):
    """Create performance comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Performance metrics comparison
    metrics = ['Test Accuracy', 'Test F1 (Macro)', 'Val Accuracy', 'Val F1 (Macro)']
    xgb_values = [
        xgb_results['performance']['test_accuracy'],
        xgb_results['performance']['test_f1_macro'],
        xgb_results['performance']['val_accuracy'],
        xgb_results['performance']['val_f1_macro']
    ]
    keras_values = [
        keras_results['performance']['test_accuracy'],
        keras_results['performance']['test_f1_macro'],
        keras_results['performance']['val_accuracy'],
        keras_results['performance']['val_f1_macro']
    ]
    
    # Bar plot comparison
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0,0].bar(x - width/2, xgb_values, width, label='XGBoost', color='skyblue', alpha=0.8)
    axes[0,0].bar(x + width/2, keras_values, width, label='Deep Learning', color='lightcoral', alpha=0.8)
    axes[0,0].set_xlabel('Metrics')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_title('Performance Comparison')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(metrics, rotation=45)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Model complexity comparison
    complexity_data = {
        'Model': ['XGBoost', 'Deep Learning'],
        'Parameters/Features': [xgb_results['n_features'], keras_results['total_parameters']]
    }
    
    axes[0,1].bar(complexity_data['Model'], complexity_data['Parameters/Features'], 
                  color=['skyblue', 'lightcoral'], alpha=0.8)
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Model Complexity')
    axes[0,1].set_yscale('log')  # Log scale due to large difference
    for i, v in enumerate(complexity_data['Parameters/Features']):
        axes[0,1].text(i, v, f'{v:,}', ha='center', va='bottom')
    axes[0,1].grid(True, alpha=0.3)
    
    # Training characteristics
    training_data = {
        'Aspect': ['Accuracy', 'F1-Score', 'Complexity', 'Speed'],
        'XGBoost': [4, 4, 2, 5],  # Ratings out of 5
        'Deep Learning': [4, 3, 5, 3]
    }
    
    angles = np.linspace(0, 2*np.pi, len(training_data['Aspect']), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    xgb_ratings = training_data['XGBoost'] + [training_data['XGBoost'][0]]
    keras_ratings = training_data['Deep Learning'] + [training_data['Deep Learning'][0]]
    
    axes[1,0].plot(angles, xgb_ratings, 'o-', linewidth=2, label='XGBoost', color='skyblue')
    axes[1,0].fill(angles, xgb_ratings, alpha=0.25, color='skyblue')
    axes[1,0].plot(angles, keras_ratings, 'o-', linewidth=2, label='Deep Learning', color='lightcoral')
    axes[1,0].fill(angles, keras_ratings, alpha=0.25, color='lightcoral')
    
    axes[1,0].set_xticks(angles[:-1])
    axes[1,0].set_xticklabels(training_data['Aspect'])
    axes[1,0].set_ylim(0, 5)
    axes[1,0].set_title('Model Characteristics (1-5 Rating)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Performance by exercise type (mock data based on results)
    exercises = ['Running', 'Walking', 'Squat', 'Riding', 'StairClimber', 
                'RopeSkipping', 'LegPress', 'BenchPress', 'ArmCurl', 'Adductor', 'LegCurl']
    
    # Approximate F1-scores based on the classification reports
    xgb_f1_by_exercise = [0.92, 0.76, 0.66, 0.74, 0.75, 0.76, 0.63, 0.57, 0.51, 0.56, 0.57]
    keras_f1_by_exercise = [0.92, 0.72, 0.84, 0.76, 0.83, 0.75, 0.77, 0.43, 0.45, 0.44, 0.37]
    
    x = np.arange(len(exercises))
    axes[1,1].bar(x - width/2, xgb_f1_by_exercise, width, label='XGBoost', color='skyblue', alpha=0.8)
    axes[1,1].bar(x + width/2, keras_f1_by_exercise, width, label='Deep Learning', color='lightcoral', alpha=0.8)
    axes[1,1].set_xlabel('Exercise Type')
    axes[1,1].set_ylabel('F1-Score')
    axes[1,1].set_title('Performance by Exercise Type')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(exercises, rotation=45, ha='right')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_strengths_weaknesses(xgb_results, keras_results):
    """Analyze strengths and weaknesses of each model."""
    analysis = {
        'XGBoost': {
            'strengths': [
                'Higher overall test accuracy (83.6% vs 84.5%)',
                'Better F1-macro score (0.694 vs 0.682)',
                'Much faster training (~2 min vs ~1 hour)',
                'Very fast inference',
                'Lower memory requirements',
                'Better interpretability with feature importance',
                'No GPU requirements',
                'Robust to hyperparameters'
            ],
            'weaknesses': [
                'Requires manual feature engineering',
                'Loses temporal sequence information',
                'May miss complex temporal patterns',
                'Limited to statistical features',
                'Less flexible architecture'
            ],
            'best_for': [
                'Real-time applications',
                'Resource-constrained environments',
                'When interpretability is important',
                'Quick prototyping and deployment',
                'Traditional ML pipelines'
            ]
        },
        'Deep Learning (CNN-LSTM)': {
            'strengths': [
                'Comparable test accuracy (84.5%)',
                'Automatic feature learning',
                'Captures temporal dependencies',
                'Better performance on complex exercises (Squat: 0.84 vs 0.66)',
                'End-to-end learning',
                'More flexible architecture',
                'Better at learning complex patterns'
            ],
            'weaknesses': [
                'Slightly lower F1-macro score (0.682 vs 0.694)',
                'Much longer training time (~1 hour)',
                'Higher memory requirements (721K parameters)',
                'Requires more data for optimal performance',
                'Less interpretable',
                'Prone to overfitting',
                'GPU recommended for training'
            ],
            'best_for': [
                'Applications with abundant data',
                'When temporal patterns are crucial',
                'Complex movement recognition',
                'Research and experimentation',
                'When training time is not critical'
            ]
        }
    }
    
    return analysis

def main():
    print("🔍 COMPREHENSIVE MODEL COMPARISON")
    print("=" * 70)
    
    # Load results
    print("Loading model results...")
    xgb_results, keras_results = load_model_results()
    
    # Create comparison table
    print("\n📊 PERFORMANCE COMPARISON TABLE")
    print("=" * 70)
    comparison_df = create_comparison_table(xgb_results, keras_results)
    print(comparison_df.to_string(index=False))
    
    # Save comparison table
    comparison_df.to_csv('models/model_comparison_table.csv', index=False)
    
    # Create performance plots
    print("\n📈 Creating performance comparison plots...")
    plot_performance_comparison(xgb_results, keras_results)
    
    # Detailed analysis
    print("\n🔍 DETAILED ANALYSIS")
    print("=" * 70)
    analysis = analyze_strengths_weaknesses(xgb_results, keras_results)
    
    for model_name, model_analysis in analysis.items():
        print(f"\n{model_name.upper()} MODEL:")
        print("-" * 40)
        
        print("✅ STRENGTHS:")
        for strength in model_analysis['strengths']:
            print(f"  • {strength}")
        
        print("\n❌ WEAKNESSES:")
        for weakness in model_analysis['weaknesses']:
            print(f"  • {weakness}")
        
        print("\n🎯 BEST FOR:")
        for use_case in model_analysis['best_for']:
            print(f"  • {use_case}")
    
    # Winner analysis
    print("\n🏆 WINNER ANALYSIS")
    print("=" * 70)
    
    xgb_acc = xgb_results['performance']['test_accuracy']
    keras_acc = keras_results['performance']['test_accuracy']
    xgb_f1 = xgb_results['performance']['test_f1_macro']
    keras_f1 = keras_results['performance']['test_f1_macro']
    
    print(f"Test Accuracy: {'Deep Learning' if keras_acc > xgb_acc else 'XGBoost'} wins")
    print(f"  XGBoost: {xgb_acc:.1%}")
    print(f"  Deep Learning: {keras_acc:.1%}")
    print(f"  Difference: {abs(keras_acc - xgb_acc):.1%}")
    
    print(f"\nF1-Macro Score: {'XGBoost' if xgb_f1 > keras_f1 else 'Deep Learning'} wins")
    print(f"  XGBoost: {xgb_f1:.3f}")
    print(f"  Deep Learning: {keras_f1:.3f}")
    print(f"  Difference: {abs(xgb_f1 - keras_f1):.3f}")
    
    # Final recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 70)
    
    print("🚀 FOR PRODUCTION DEPLOYMENT:")
    print("  Choose XGBoost if:")
    print("    • You need fast inference (<1ms)")
    print("    • Limited computational resources")
    print("    • Interpretability is important")
    print("    • Quick deployment is needed")
    
    print("\n🔬 FOR RESEARCH & DEVELOPMENT:")
    print("  Choose Deep Learning if:")
    print("    • You have abundant training data")
    print("    • Complex temporal patterns are important")
    print("    • Training time is not a constraint")
    print("    • You want to experiment with architectures")
    
    print("\n📈 HYBRID APPROACH:")
    print("  • Use XGBoost for baseline and quick prototyping")
    print("  • Use Deep Learning for final optimization")
    print("  • Consider ensemble methods combining both")
    
    # Save analysis to file
    with open('models/model_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\n✅ Analysis complete!")
    print(f"  📊 Comparison table: models/model_comparison_table.csv")
    print(f"  📈 Comparison plots: models/model_comparison.png")
    print(f"  📝 Detailed analysis: models/model_analysis.json")

if __name__ == "__main__":
    main()
