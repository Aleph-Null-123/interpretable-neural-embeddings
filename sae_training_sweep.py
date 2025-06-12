from sae_training_utils import run_all_saes

rats = ['gatsby', 'cicero', 'buddy', 'achilles']

embedding_dimensions = [8, 16, 32, 64]

for name in rats:
    for embedding_dimension in embedding_dimensions:
        run_all_saes(
            emb_path=f"final_embeddings/output_dimension_{embedding_dimension}_{name}/train_embeddings.npy",
            save_dir=f"sae_gridsearch_2/{name}_{embedding_dimension}/",
            architectures=["standard", "topk", "jumprelu"],
            expansion_factors=[2, 4, 8],
            lr_vals=[1e-4, 3e-4, 1e-3],
            l1_vals=[1e-4, 1e-3, 1e-2],
            topk_vals=[8, 16, 32],
            bw_vals=[5e-4, 1e-3, 5e-3],
            epochs=50
        )