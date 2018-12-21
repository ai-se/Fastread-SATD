from sklearn.model_selection import train_test_split

from loader import SATDD, DATASET
from processor import k_fold_with_tuning
import logging

from utility import process_output, compare_with_huang

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logs/12_20_18_test.log', level=logging.DEBUG, format='%(asctime)s %(message)s')
import multiprocessing as mp
from joblib import Parallel, delayed


MAX_FEATURE = .5
STOP_WORDS = "english"

FOLD = 5
POOL_SIZE = 100000
INIT_POOL_SIZE = 10
BUDGET = 10

SINGLE_PROJECT_FOLD = 5
SINGLE_PROJECT_FRAC = 0.3


def run_rig(filename):
    logger.info("\n++++++++++NEW RIG++++++++++\n")

    satdd = SATDD()
    satdd = satdd.load_data(filename)

    #make_results(satdd)

    cross_project(satdd)


def single_project(satdd):
    all_datasets = satdd.all_dataset_pd.projectname.unique()
    logger.info("=====SINGLE PROJECT=====")
    num_cpu = mp.cpu_count()

    Parallel(n_jobs=(num_cpu-3))(delayed(run_rig_on_single_project)(satdd, dataset) for dataset in all_datasets)

def run_rig_on_single_project(satdd, project_name):
    project_data = satdd.create_and_process_dataset([project_name],
                                                     doInclude=True)

    logger.info("DATASET: " + project_name + " | MAX_FEATURE: " + str(MAX_FEATURE) + " | STOP_WORDS: " + str(STOP_WORDS)
                + " | FOLD: " + str(FOLD) + " | POOL_SIZE: " + str(POOL_SIZE) + " | INIT_POOL: "
                + str(INIT_POOL_SIZE) + " | BUDGET: " + str(BUDGET)
                + " | SINGLE_PROJ_FOLD: " + str(SINGLE_PROJECT_FOLD))

    for i in range(SINGLE_PROJECT_FOLD):
        project_data.data_pd = project_data.data_pd.sample(frac=1)
        train, test = train_test_split(project_data.data_pd, test_size=SINGLE_PROJECT_FRAC)

        training_data = DATASET(train)
        test_data = DATASET(test)

        # Logging Ground Truth
        logger.info(project_name + " | FOLD: " + str(i) + " | FRAC: " + str(SINGLE_PROJECT_FRAC))
        logger.info(project_name + " | TRAINING DATA: TRUE: " + str(training_data.true_count) + " | FALSE: "
                    + str(training_data.false_count))
        logger.info(project_name + " | TEST DATA: TRUE: " + str(test_data.true_count) + " | FALSE: "
                    + str(test_data.false_count))

        # no need to give a tfidf, will calculate itself
        training_data.set_csr_mat(max_f=MAX_FEATURE, stop_w=STOP_WORDS)
        # need to give the tfidf from training set, will just use transform to create csr_matrix
        test_data.set_csr_mat(max_f=MAX_FEATURE, stop_w=STOP_WORDS, tfer=training_data.tfer)

        conf_mat, clfs = k_fold_with_tuning(test_data, training_data, fold=FOLD, pool_size=POOL_SIZE,
                                            init_pool=INIT_POOL_SIZE, budget=BUDGET, label=project_name)

        print(conf_mat)

        print(project_name + " END FOLD " + str(i))

    print("END")







def cross_project(satdd):
    all_datasets = satdd.all_dataset_pd.projectname.unique()
    logger.info("=====CROSS PROJECT=====")

    num_cpu = mp.cpu_count()

    Parallel(n_jobs=1)(delayed(run_rig_on_project)(satdd, dataset) for dataset in all_datasets)

    # for dataset in all_datasets:
    #     if 'apache-ant-1.7.0' in dataset or 'emf-2.4.1' in dataset:
    #         continue
    #
    #     run_rig_on_project(satdd, dataset)


def run_rig_on_project(satdd, project_name):
    training_data = satdd.create_and_process_dataset([project_name],
                                                     doInclude=False)
    # no need to give a tfidf, will calculate itself
    training_data.set_csr_mat(max_f=MAX_FEATURE, stop_w=STOP_WORDS)
    test_data = satdd.create_and_process_dataset([project_name], doInclude=True)
    # need to give the tfidf from training set, will just use transform to create csr_matrix
    test_data.set_csr_mat(max_f=MAX_FEATURE, stop_w=STOP_WORDS, tfer=training_data.tfer)

    # Logging rig descriptions
    logger.info("\n===============")
    logger.info("DATASET: " + project_name + " | MAX_FEATURE: " + str(MAX_FEATURE) + " | STOP_WORDS: " + str(STOP_WORDS)
                + " | FOLD: " + str(FOLD) + " | POOL_SIZE: " + str(POOL_SIZE) + " | INIT_POOL: "
                + str(INIT_POOL_SIZE) + " | BUDGET: " + str(BUDGET))

    # Logging Ground Truth
    logger.info("TRAINING DATA: TRUE: " + str(training_data.true_count) + " | FALSE: "
                + str(training_data.false_count))
    logger.info("TEST DATA: TRUE: " + str(test_data.true_count) + " | FALSE: "
                + str(test_data.false_count))

    conf_mat, clfs = k_fold_with_tuning(test_data, training_data, fold=FOLD, pool_size=POOL_SIZE,
                                        init_pool=INIT_POOL_SIZE, budget=BUDGET, label=project_name)

    try:
        with open("conf_mat.txt", "a+") as f:
            f.write(conf_mat)
            f.write("+++++++++++++++++++++")
    except:
        print(str(conf_mat))
    logger.info("END CROSS PROJECT=====")




def make_results(satdd):
    compare_with_huang("results/12_21_18/cross_proj_avg_f.txt", "results/12_21_18_cross_fea30_stopEng_f.csv")
    #process_output(satdd, '12_21_18_cross_fea30.log', "12_21_18/")


