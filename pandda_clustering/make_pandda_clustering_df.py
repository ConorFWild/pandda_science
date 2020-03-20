
if __name__ == "__main__":
    args = parse_args()

    config = get_config(args)

    output: Output = setup_output_directory(config.our_dir_path)

    embedding_dfs = get_embedding_dfs()

    summary_df = summarise_embedding_dfs(embedding_dfs)

    output_summary_df(summary_df,
                      output.summary_df_path,
                      )