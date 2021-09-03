import libs.parameters as parameters
import json
import sys
import os
from main import main
import shutil
import tools.evaluate as evalaute
import tools.surrogate_mapping as surrogate_mapping


def create_surrogate_mapping(data_dir, g_config, seed):
    """
    In case of SiameseXML: it'll just remove invalid labels
    However, keeping this code as user might want to try out 
    alternate mappings as well
    
    ##FIXME: For non-shared vocabulary
    """
    dataset = g_config['dataset']
    try:
        surrogate_threshold = g_config['surrogate_threshold']
        surrogate_method = g_config['surrogate_method']
    except KeyError:
        surrogate_threshold = -1
        surrogate_method = 0
    
    arch = g_config['arch']
    tmp_model_dir = os.path.join(
        data_dir, dataset, f'siamesexml.{arch}', f"{surrogate_threshold}.{seed}")
    data_dir = os.path.join(data_dir, dataset)
    try:
        os.makedirs(tmp_model_dir, exist_ok=False)
        surrogate_mapping.run(
            feat_fname=os.path.join(data_dir, g_config["trn_feat_fname"]),
            lbl_feat_fname=os.path.join(data_dir, g_config["lbl_feat_fname"]),
            lbl_fname=os.path.join(data_dir, g_config["trn_label_fname"]),
            feature_type=g_config["feature_type"],
            method=surrogate_method,
            threshold=surrogate_threshold,
            seed=seed,
            tmp_dir=tmp_model_dir)
    except FileExistsError:
        print("Using existing data for surrogate task!")
    finally:
        data_stats = json.load(
            open(os.path.join(tmp_model_dir, "data_stats.json")))
        mapping = os.path.join(
            tmp_model_dir, 'surrogate_mapping.txt')
    return data_stats, mapping


def evaluate(g_config, data_dir, pred_fname, filter_fname=None, betas=-1, n_learners=1):
    if n_learners == 1:
        func = evalaute.main
    else:
        raise NotImplementedError("")

    dataset = g_config['dataset']
    data_dir = os.path.join(data_dir, dataset)
    A = g_config['A']
    B = g_config['B']
    top_k = g_config['top_k']
    ans = func(
        tst_label_fname=os.path.join(
            data_dir, g_config["tst_label_fname"]),
        trn_label_fname=os.path.join(
            data_dir, g_config["trn_label_fname"]),
        pred_fname=pred_fname,
        A=A, 
        B=B,
        filter_fname=filter_fname, 
        betas=betas, 
        top_k=top_k,
        save=g_config["save_predictions"])
    return ans


def print_run_stats(train_time, model_size, avg_prediction_time, fname=None):
    line = "-"*30 
    out = f"Training time (sec): {train_time:.2f}\n"
    out += f"Model size (MB): {model_size:.2f}\n"
    out += f"Avg. Prediction time (msec): {avg_prediction_time:.2f}"
    out = f"\n\n{line}\n{out}\n{line}\n\n"
    print(out)
    if fname is not None:
        with open(fname, "a") as fp:
            fp.write(out)


def run_siamesexml(work_dir, pipeline, version, seed, config):

    # fetch arguments/parameters like dataset name, A, B etc.
    g_config = config['global']
    dataset = g_config['dataset']
    arch = g_config['arch']

    # run stats
    train_time = 0
    model_size = 0
    avg_prediction_time = 0

    # Directory and filenames
    data_dir = os.path.join(work_dir, 'data')

    filter_fname = os.path.join(data_dir, dataset, 'filter_labels_test.txt')
    if not os.path.isfile(filter_fname):
        filter_fname = None
    
    result_dir = os.path.join(
        work_dir, 'results', pipeline, arch, dataset, f'v_{version}')
    model_dir = os.path.join(
        work_dir, 'models', pipeline, arch, dataset, f'v_{version}')
    _args = parameters.Parameters("Parameters")
    _args.parse_args()
    _args.update(config['global'])
    _args.update(config['siamese'])
    _args.params.seed = seed

    args = _args.params
    args.data_dir = data_dir
    args.model_dir = os.path.join(model_dir, 'siamese')
    args.result_dir = os.path.join(result_dir, 'siamese')

    # Create the label mapping for classification surrogate task
    data_stats, args.surrogate_mapping = create_surrogate_mapping(
        data_dir, g_config, seed)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # train intermediate representation
    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['surrogate'].split(",")
    args.num_labels = int(temp[2])

    ##FIXME: For non-shared vocabulary

    args.vocabulary_dims_document = int(temp[0])
    args.vocabulary_dims_label = int(temp[0])
    _train_time, _ = main(args)
    train_time += _train_time

    # train final representation and extreme classifiers
    _args.update(config['extreme'])
    args = _args.params
    args.surrogate_mapping = None
    args.model_dir = os.path.join(model_dir, 'extreme')
    args.result_dir = os.path.join(result_dir, 'extreme')
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    args.mode = 'train'
    args.arch = os.path.join(os.getcwd(), f'{arch}.json')
    temp = data_stats['extreme'].split(",")
    args.num_labels = int(temp[2])
    args.vocabulary_dims = int(temp[0])
    _train_time, _model_size = main(args)
    train_time += _train_time
    model_size += _model_size

    # predict using extreme classifiers
    args.pred_fname = 'tst_predictions'
    args.mode = 'predict'
    _, _, _pred_time = main(args)
    avg_prediction_time += _pred_time

    # copy the prediction files to level-1
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_clf.npz'),
        os.path.join(result_dir, 'tst_predictions_clf.npz'))
    shutil.copy(
        os.path.join(result_dir, 'extreme', 'tst_predictions_knn.npz'),
        os.path.join(result_dir, 'tst_predictions_knn.npz'))

    # evaluate
    pred_fname = os.path.join(result_dir, 'tst_predictions')
    ans = evaluate(
        g_config=g_config,
        data_dir=data_dir,
        pred_fname=pred_fname,
        filter_fname=filter_fname,
        betas=[0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    print(ans)
    f_rstats = os.path.join(result_dir, 'log_eval.txt')
    with open(f_rstats, "w") as fp:
        fp.write(ans)

    print_run_stats(train_time, model_size, avg_prediction_time, f_rstats)
    return os.path.join(result_dir, f"score_{g_config['beta']:.2f}.npz"), \
        train_time, model_size, avg_prediction_time


if __name__ == "__main__":
    pipeline = sys.argv[1]
    work_dir = sys.argv[2]
    version = sys.argv[3]
    config = sys.argv[4]
    seed = int(sys.argv[5])
    if pipeline == "SiameseXML" or pipeline == "SiameseXML++":
        run_siamesexml(
            pipeline=pipeline,
            work_dir=work_dir,
            version=f"{version}_{seed}",
            seed=seed,
            config=json.load(open(config)))
    else:
        raise NotImplementedError("")
