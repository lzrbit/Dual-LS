import os


def create_dir(args):
    exp_dir = f"{args.root_dir}result_{args.experiment_index}"
    if not os.path.exists(exp_dir):
        # Create the base directory
        os.makedirs(exp_dir)
        print(f"Created directory: {exp_dir}")
        result_dir = os.path.join(exp_dir, 'log')
        saved_dir = os.path.join(exp_dir, 'weights')
        os.makedirs(result_dir)
        print(f"Created subdirectory: {result_dir}")
        os.makedirs(saved_dir)
        print(f"Created subdirectory: {saved_dir}")
    else:
        result_dir = os.path.join(exp_dir, 'log')
        saved_dir = os.path.join(exp_dir, 'weights')
        print(f"Directory {exp_dir} already exists. No new directories were created.")
    return result_dir, saved_dir