python3.7m ihead_basic_main.py max_iters=5000 log_probes=True eval_delta=5 loss_head_only=True \
        data_args.k=1 data_args.fixed_special_toks=True \
        optim_args.use_sgd=True optim_args.learning_rate=0.03 optim_args.weight_decay=1e-4 optim_args.batch_size=512 \
        model_args.first_ffn=True model_args.linear_first_ffn=False \
        model_args.final_ffn=True model_args.linear_final_ffn=False \
        noise_prob=0.5 no_residual=False \
        model_args.freeze_embeddings=True model_args.freeze_output=True model_args.dim=256 \
        save_model=True save_model_dir=k1_all_ffn_noise0.5_all_train_withresidual_firstRandom_run2.model
