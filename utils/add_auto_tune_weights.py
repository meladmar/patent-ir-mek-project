# ========================
# ✅ Auto Tune Weights
# ========================
def tune_weights():
    results = {
        'alpha': [],
        'beta': [],
        'NDCG@10': [],
        'MAP@100': [],
        'Recall@100': []
    }

    for alpha in np.arange(0.4, 0.71, 0.05):
        beta = round(1 - alpha, 2)
        metrics = run_hybrid_eval(
            variant='title_abstract_claims',
            output_csv=f'/home/lucid/Desktop/eval/hybrid_a{alpha}_b{beta}.csv',
            alpha=alpha,
            beta=beta
        )
        results['alpha'].append(alpha)
        results['beta'].append(beta)
        results['NDCG@10'].append(metrics['ndcg_cut_10'])
        results['MAP@100'].append(metrics['map_cut_100'])
        results['Recall@100'].append(metrics['recall_100'])

    df = pd.DataFrame(results)
    df.to_csv('/home/lucid/Desktop/eval/tuning_summary.csv', index=False)
    df.set_index('alpha')[['NDCG@10', 'MAP@100', 'Recall@100']].plot(
        title="Metric Trends vs Alpha (BM25 weight)",
        marker='o', figsize=(10,5)
    )
    plt.grid(True)
    plt.savefig('/home/lucid/Desktop/eval/tuning_plot.png')
    plt.show()

# ========================
# ✅ Entry Point
# ========================
if __name__ == '__main__':
    tune_weights()

