
# (c) 2019 Horie Yuki

from __future__ import print_function
import os

import sys
import random
import math
import time
import copy

import itertools
from pathlib import Path
from inspect import currentframe

import numpy as np
import numpy
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pydotplus

from PIL import ImageTk, Image

import tkinter
import tkinter.filedialog
from tkinter import ttk,N,E,S,W,font

import sklearn
import sklearn.ensemble

from sklearn import linear_model
from sklearn import svm
from sklearn.externals.six import StringIO
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import _tree

import GPy
import GPyOpt

import category_encoders

try:
    import dtreeviz.trees
    import dtreeviz.shadow
    import dtreeviz
except:
    print('dtreeviz was not found')
    pass

import xgboost as xgb
#from xgboost import plot_tree     #sklearn has also plot_tree, so dont import plot_Tree

import optuna

# #chkprint
# refer from https://qiita.com/AnchorBlues/items/f7725ba87ce349cb0382
def chkprint(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    print(', '.join(names.get(id(arg),'???')+' = '+repr(arg) for arg in args))

# get variable name
def get_variablename(*args):
    names = {id(v):k for k,v in currentframe().f_back.f_locals.items()}
    return '_' + ', '.join(names.get(id(arg),'???') + '_' + repr(arg) for arg in args)

# fix the random.seed, it can get the same results every time to run this program
np.random.seed(1)
random.seed(1)

current_path = Path.cwd()
program_path = Path(__file__).parent.resolve()
parent_path = program_path.parent.resolve()

data_path           = parent_path / 'data'
data_processed_path = data_path / 'processed'
graphviz_path = program_path / 'release' / 'bin' / 'dot.exe'
graphviz_path = graphviz_path.resolve()
chkprint(program_path)

columns_results = [ 'model_name',
                    'train_model_mse',
                    'train_model_rmse',
                    'test_model_mse',
                    'test_model_rmse',
                    'train_model_score',
                    'test_model_score']

allmodel_results_raw_df = pd.DataFrame(columns = columns_results)
allmodel_results_std_df = pd.DataFrame(columns = columns_results)

summary_results_raw_df = pd.DataFrame(columns = columns_results)
summary_results_std_df = pd.DataFrame(columns = ['model_name', 'train_model_std_mse',  'test_model_std_mse'])


def learning():
    '''
    if Booleanvar_bayesian_opt.get() == True and t_output_clm_num != 1:
        from tkinter import messagebox
        messagebox.showwarning('Warning', 'ベイズ最適化は目的関数が１つでのみ有効です')
        return
    '''
    # cav & theme
    csv_path = t_csv_filepath.get()

    # theme_name used as the result folder name
    theme_name = t_theme_name.get()

    # evaluate of all candidate or not
    is_gridsearch = Booleanvar_gridsearch.get()

    # perform deeplearning or not
    is_dl = Booleanvar_deeplearning.get()
    is_bayesian_opt = Booleanvar_bayesian_opt.get()
    is_optuna_sklearn = Booleanvar_optuna_sklearn.get()
    is_optuna_deeplearning = Booleanvar_optuna_deeplearning.get()

    # make the save folder
    print('change directory to : ', os.path.dirname(csv_path))
    os.chdir(os.path.dirname(csv_path))

    # make the folder for saving the results

    paths = [ parent_path / 'results' / theme_name / 'tree',
              parent_path / 'results' / theme_name / 'importance',
              parent_path / 'results' / theme_name / 'parameter_raw',
              parent_path / 'results' / theme_name / 'parameter_std',
              parent_path / 'results' / theme_name / 'predict_raw',
              parent_path / 'results' / theme_name / 'predict_sr_contour',
              parent_path / 'results' / theme_name / 'predict_stdtoraw',
              parent_path / 'results' / theme_name / 'traintest_raw',
              parent_path / 'results' / theme_name / 'traintest_std',
              parent_path / 'results' / theme_name / 'traintest_stdtoraw',
              parent_path / 'results' / theme_name / 'scatter_diagram',
              parent_path / 'results' / theme_name / 'bayesian_opt',
              parent_path / 'results' / theme_name / 'DL-models']

    for path_name in paths:
        os.makedirs(path_name, exist_ok=True)

    # csv information
    info_num    = int(t_info_clm_num.get())     # information columns in csv file
    input_num   = int(t_input_clm_num.get())    # input data  columns in csv file
    output_num  = int(t_output_clm_num.get())   # output data columns in csv file

    try:
        raw_data_df = pd.read_csv(open(csv_path ,encoding="utf-8_sig"))
        print('utf-8で読み込みました')
    except:
        raw_data_df = pd.read_csv(open(csv_path ,encoding="shift-jis"))
        print('shift-jisで読み込みました')
    else:
        pass

    # csv information data columns
    raw_data_info_col    = info_num
    raw_data_input_col   = info_num + input_num
    raw_data_output_col  = info_num + input_num + output_num

    info_raw_df         = raw_data_df.iloc[:, 0 : raw_data_info_col]
    info_processed_df   = info_raw_df
    in_output_raw_df    = raw_data_df.iloc[:, raw_data_info_col  : raw_data_output_col]

    # one-hot vectorize
    category_data_df = in_output_raw_df.select_dtypes(exclude=['number', 'bool'])
    category_list = list(category_data_df.columns)
    ce_onehot   = category_encoders.OneHotEncoder(cols = category_list, handle_unknown = 'impute')
    processed_data_df = ce_onehot.fit_transform(in_output_raw_df)

    input_num_plus= len(processed_data_df.columns) - len(in_output_raw_df.columns)

    input_num   += input_num_plus
    list_num    = [input_num, output_num]

    info_col    = info_num
    input_col   = info_num + input_num
    output_col  = info_num + input_num + output_num

    print('processed_data_df.head()')
    print(processed_data_df.head())
    print('processed_data_df.describe()')
    print(processed_data_df.describe())

    input_processed_df          = processed_data_df.iloc[:, info_col  : input_col]
    output_processed_df         = processed_data_df.iloc[:, input_col : output_col]
    lastoutput_processed_df    =  processed_data_df.iloc[:, -1:]

    #print(lastoutput_processed_df)

    in_output_processed_df = pd.concat([input_processed_df, output_processed_df], axis=1)
    list_df             = [input_processed_df, output_processed_df]

    info_feature_names              = info_processed_df.columns
    input_feature_names             = input_processed_df.columns
    output_feature_names            = output_processed_df.columns
    lastoutput_feature_names        = lastoutput_processed_df.columns
    list_feature_names              = [input_feature_names, output_feature_names]

    predict_input_feature_names     = list(map(lambda x:x + '-predict' , input_feature_names))
    predict_output_feature_names    = list(map(lambda x:x + '-predict' , output_feature_names))
    predict_lastoutput_feature_names    = list(map(lambda x:x + '-predict' , lastoutput_feature_names))

    list_predict_feature_names      = [predict_input_feature_names, predict_output_feature_names]

    in_output_sc_model  = StandardScaler()
    input_sc_model      = StandardScaler()
    output_sc_model     = StandardScaler()
    lastoutput_sc_model = StandardScaler()
    list_sc_model       = [input_sc_model, output_sc_model]

    in_output_std_df    = pd.DataFrame(in_output_sc_model.fit_transform(in_output_processed_df))
    input_std_df        = pd.DataFrame(input_sc_model.fit_transform(input_processed_df))
    output_std_df       = pd.DataFrame(output_sc_model.fit_transform(output_processed_df))
    lastoutput_std_df   = pd.DataFrame(lastoutput_sc_model.fit_transform(lastoutput_processed_df))

    input_raw_des   = input_processed_df.describe()
    output_raw_des  = output_processed_df.describe()

    input_raw_max   = input_raw_des.loc['max']
    input_raw_min   = input_raw_des.loc['min']
    input_raw_mean  = input_raw_des.loc['mean']
    output_raw_max  = output_raw_des.loc['max']
    output_raw_min  = output_raw_des.loc['min']
    output_raw_mean = output_raw_des.loc['mean']

    list_raw_max    = [input_raw_max, output_raw_max]
    list_raw_min    = [input_raw_min, output_raw_min]
    list_raw_mean   = [input_raw_mean, output_raw_mean]

    input_std_des   = input_std_df.describe()
    output_std_des  = output_std_df.describe()

    input_std_max   = input_std_des.loc['max']
    input_std_min   = input_std_des.loc['min']
    output_std_max  = output_std_des.loc['max']
    output_std_min  = output_std_des.loc['min']

    list_std_max    = [input_std_max, output_std_max]
    list_std_min    = [input_std_min, output_std_min]

    # split train data and test data from the in_output_std_df
    np.random.seed(10)
    random.seed(10)
    train_std_df, test_std_df  = train_test_split(in_output_std_df, test_size=0.1)
    train_std_df, val_std_df   = train_test_split(train_std_df, test_size=0.11111)

    np.random.seed(10)
    random.seed(10)
    train_raw_df, test_raw_df  = train_test_split(in_output_processed_df, test_size=0.1)
    train_raw_df, val_raw_df   = train_test_split(train_raw_df, test_size=0.11111)

    # transform from pandas dataframe to numpy array
    train_raw_np = np.array(train_raw_df)
    test_raw_np  = np.array(test_raw_df)
    val_raw_np   = np.array(val_raw_df)

    train_std_np = np.array(train_std_df)
    test_std_np  = np.array(test_std_df)
    val_std_np   = np.array(val_std_df)

    print(len(train_raw_np))
    print(len(test_raw_np))
    print(len(val_raw_np))

    # split columns to info, input, output
    [train_input_raw, train_output_raw] = np.hsplit(train_raw_np, [input_num])

    list_train_raw                      = [train_input_raw, train_output_raw]
    [test_input_raw,  test_output_raw]  = np.hsplit(test_raw_np,  [input_num])
    list_test_raw                       = [test_input_raw, test_output_raw]
    [val_input_raw,  val_output_raw]    = np.hsplit(val_raw_np,  [input_num])
    list_val_raw                        = [val_input_raw, val_output_raw]

    [train_input_std, train_output_std] = np.hsplit(train_std_np, [input_num])
    list_train_std                      = [train_input_std  , train_output_std]
    [test_input_std,  test_output_std]  = np.hsplit(test_std_np , [input_num])
    list_test_std                       = [test_input_std   , test_output_std]
    [val_input_std,  val_output_std]    = np.hsplit(val_std_np,  [input_num])
    list_val_std                        = [val_input_std, val_output_std]

    train_input_raw_df                  = pd.DataFrame(train_input_raw  , columns = input_feature_names)
    test_input_raw_df                   = pd.DataFrame(test_input_raw   , columns = input_feature_names)
    val_input_raw_df                    = pd.DataFrame(val_input_raw   , columns = input_feature_names)
    train_output_raw_df                 = pd.DataFrame(train_output_raw , columns = output_feature_names)
    test_output_raw_df                  = pd.DataFrame(test_output_raw  , columns = output_feature_names)
    val_output_raw_df                   = pd.DataFrame(val_output_raw  , columns = output_feature_names)

    list_train_raw_df                   = [train_input_raw_df, train_output_raw_df]
    list_test_raw_df                    = [test_input_raw_df, test_output_raw_df]
    list_val_raw_df                     = [val_input_raw_df, val_output_raw_df]

    train_input_std_df                  = pd.DataFrame(train_input_std  , columns = input_feature_names)
    test_input_std_df                   = pd.DataFrame(test_input_std   , columns = input_feature_names)
    val_input_std_df                    = pd.DataFrame(val_input_std   , columns = input_feature_names)
    train_output_std_df                 = pd.DataFrame(train_output_std , columns = output_feature_names)
    test_output_std_df                  = pd.DataFrame(test_output_std  , columns = output_feature_names)
    val_output_std_df                   = pd.DataFrame(val_output_std  , columns = output_feature_names)

    list_train_std_df                   = [train_input_std_df , train_output_std_df ]
    list_test_std_df                    = [test_input_std_df  , test_output_std_df ]
    list_val_std_df                     = [val_input_std_df  , val_output_std_df ]

    print('The number of train-test-validation')
    print('train : ',len(train_input_std_df))
    print('test : ',len(test_input_std_df))
    print('val : ',len(val_input_std_df))
    print('')

    plt.figure(figsize=(5,5))
    sns.heatmap(in_output_raw_df.corr(), cmap = 'Oranges',annot=False, linewidths = .5)
    plt.savefig(parent_path / 'results' /  theme_name / 'correlation_coefficient.png', dpi = 240)
    #plt.close()

    plt.figure(figsize=(5,5))
    sns.pairplot(in_output_raw_df)
    plt.savefig(parent_path / 'results' /  theme_name / 'pairplot.png', dpi = 240)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img4 = Image.open(parent_path / 'results' / theme_name / 'pairplot.png')
    img4_resize = img4.resize((photo_size, photo_size), Image.LANCZOS)
    img4_resize.save(parent_path / 'results' / theme_name / 'pairplot_resized.png')

    global image_pairplot
    image_open = Image.open(parent_path / 'results' / theme_name / 'pairplot_resized.png')
    image_pairplot = ImageTk.PhotoImage(image_open, master=frame2)

    canvas_pairplot.create_image(int(photo_size/2),int(photo_size/2), image=image_pairplot)

    #######  extract descision tree   ################################################
    def tree_to_code(tree, feature_names):
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        #print("def tree({}):".format(", ".join(feature_names)))

        def recurse(node, depth):
            indent = "  " * depth

            with open(str(save_address)+ 'tree.txt', 'a') as f:
                if tree_.feature[node] != _tree.TREE_UNDEFINED:
                    name = feature_name[node]
                    threshold = tree_.threshold[node]
                    recurse(tree_.children_left[node], depth + 1)
                    recurse(tree_.children_right[node], depth + 1)
                else:
                    pass

        recurse(0, 1)

    def gridsearch_predict(model_raw,model_std, model_name):
        start_time = time.time()

        gridsearch_predict_raw_df   = pd.DataFrame(model_raw.predict(gridsearch_input_raw), columns = list_predict_feature_names[out_n])
        gridsearch_predict_std_df   = pd.DataFrame(model_std.predict(gridsearch_input_std), columns = list_predict_feature_names[out_n])
        gridsearch_predict_stdtoraw_df       = pd.DataFrame(list_sc_model[out_n].inverse_transform(gridsearch_predict_std_df), columns = list_predict_feature_names[out_n])

        end_time = time.time()
        total_time = end_time - start_time
        gridsearch_predict_raw_df       = pd.concat([gridsearch_input_raw_df, gridsearch_predict_raw_df], axis = 1)
        gridsearch_predict_stdtoraw_df  = pd.concat([gridsearch_input_std_df, gridsearch_predict_stdtoraw_df], axis = 1)

        gridsearch_predict_raw_df.to_csv(     parent_path / 'results' / theme_name / 'predict_raw' / (str(model_name) + '_predict.csv'), index=False)
        gridsearch_predict_stdtoraw_df.to_csv(parent_path / 'results' / theme_name / 'predict_stdtoraw' / (str(model_name)+ '_predict.csv'), index=False)

        return


    #### dtree viz #####
    '''
    train_output_  = train_output.flatten()
    try:
        viz = dtreeviz.trees.dtreeviz(model_raw,
                                    train_input,
                                    train_output_,
                                    target_name   = list_predict_feature_names[out_n],
                                    feature_names = list_predict_feature_names[in_n])

        viz.save('results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'tree' + os.sep + 'decisiontree' + str(max_depth) + '.svg')
        #viz.view()
        #sys.exit()

    except:
        #print('dtreeviz error')
        # PATH
        # output_num more than 2

    '''

    #########   regression by the scikitlearn model ###############
    def fit_model_std_raw(model, model_name):
        print(model_name)
        start_time = time.time()

        model_raw = copy.deepcopy(model)
        model_std = copy.deepcopy(model)

        model_raw.fit(list_train_raw[in_n], list_train_raw[out_n])
        model_std.fit(list_train_std[in_n], list_train_std[out_n])

        model_raw, model_std = save_regression(model_raw, model_std, model_name)
        return model_raw, model_std


    def save_regression(model_raw, model_std, model_name):
        def save_tree_topdf(model, model_name):
            dot_data = StringIO()
            try:
                sklearn.tree.export_graphviz(model, out_file=dot_data, feature_names = list_feature_names[in_n])
            except:
                xgb.to_graphviz(model,  out_file=dot_data, feature_names = list_feature_names[in_n])

            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
            # refer from https://qiita.com/wm5775/items/1062cc1e96726b153e28
            # the Graphviz2.38 dot.exe
            graph.progs = {'dot':graphviz_path}

            #graph.write_pdf(model_name + '.pdf')

            #graph.write_pdf(os.path.join(str(parent_path.resolve()), 'results', theme_name, 'sklearn', 'tree', str(model_name) + '.pdf'))
            #tmp_path = (parent_path / 'results' / theme_name / 'tree' / (str(model_name) + '.pdf')).name
            #graph.write_pdf(tmp_path)
            #graph.write_pdf(r'C:\Users\1310202\Desktop\20180921\horie\data_science\データ解析\CAB-Analyze-Bigdata-master\new\src\results\Bushing\sklearn\tree\DecisionTreeRegressor_max_depth_7.pdf')

            pass

            return

        def save_importance_features(model, model_name):

            importances = pd.Series(model.feature_importances_)
            importances = np.array(importances)

            label       = list_feature_names[in_n]
            chkprint(model_name)
            chkprint(label)
            chkprint(importances)
            if inv_ == True:
                #sys.exit()
                pass

            plt.bar(label, importances)

            plt.xticks(rotation=90)
            plt.xticks(fontsize=8)
            plt.rcParams["font.size"] = 12

            plt.title("importances-" + model_name)
            plt.savefig(parent_path / 'results' / theme_name / 'importance' / (str(model_name)+ '.png'), dpi = 240)
            #plt.close()
            return

        def save_scatter_diagram(model, model_name):

            pred_train = model.predict(list_train_raw[in_n])
            pred_test = model.predict(list_test_raw[in_n])

            #print('size', len((pred_train)), len(list_train_raw[out_n]))


            plt.figure(figsize=(5,5))
            plt.scatter(list_train_raw[out_n], pred_train, label = 'Train', c = 'blue')
            plt.title("-" + model_name)
            plt.xlabel('Measured value')
            plt.ylabel('Predicted value')
            plt.scatter(list_test_raw[out_n], pred_test, c = 'lightgreen', label = 'Test', alpha = 0.8)
            plt.legend(loc = 4)
            #print('before scatter figure')
            plt.savefig(parent_path / 'results' / theme_name / 'scatter_diagram' /  (str(model_name) + '_scatter.png'))
            #print('saved scatter figure')
            #plt.close()

        global allmodel_results_raw_df
        global allmodel_results_std_df
        global allmodel_results_stdtoraw_df
        global allmodel_bayesian_opt_df

        train_output_predict_raw    = model_raw.predict(list_train_raw[in_n])
        test_output_predict_raw     = model_raw.predict(list_test_raw[in_n])
        train_output_predict_std    = model_std.predict(list_train_std[in_n])
        test_output_predict_std     = model_std.predict(list_test_std[in_n])

        train_output_predict_stdtoraw    = model_std.predict(list_train_std[in_n])
        test_output_predict_stdtoraw     = model_std.predict(list_test_std[in_n])
        train_output_predict_stdtoraw    = list_sc_model[out_n].inverse_transform(train_output_predict_std)
        test_output_predict_stdtoraw     = list_sc_model[out_n].inverse_transform(test_output_predict_std)

        if hasattr(model_raw, 'score') == True:
            train_model_raw_score   = model_raw.score(list_train_raw[in_n] , list_train_raw[out_n])
            train_model_raw_score   = np.round(train_model_raw_score, 5)
            test_model_raw_score    = model_raw.score(list_test_raw[in_n],   list_test_raw[out_n])
            test_model_raw_score    = np.round(test_model_raw_score, 5)

        if hasattr(model_raw, 'evaluate') == True:
            train_model_raw_score   = model_raw.evaluate(list_train_raw[in_n] , list_train_raw[out_n])
            train_model_raw_score   = np.round(train_model_raw_score, 5)
            test_model_raw_score    = model_raw.evaluate(test_input_raw,   list_test_raw[out_n])
            test_model_raw_score    = np.round(test_model_raw_score, 5)

        if hasattr(model_std, 'score') == True:
            train_model_std_score   = model_std.score(list_train_std[in_n] , list_train_std[out_n])
            train_model_std_score   = np.round(train_model_std_score, 5)
            test_model_std_score    = model_std.score(list_test_std[in_n],   list_test_std[out_n])
            test_model_std_score    = np.round(test_model_std_score, 5)

        if hasattr(model_std, 'evaluate') == True:
            train_model_std_score   = model_std.evaluate(list_train_std[in_n] , list_train_std[out_n])
            train_model_std_score   = np.round(train_model_std_score, 5)
            test_model_std_score    = model_std.evaluate(test_input_std,   list_test_std[out_n])
            test_model_std_score    = np.round(test_model_std_score, 5)

        train_output_predict_raw_df = pd.DataFrame(train_output_predict_raw, columns = list_predict_feature_names[out_n])
        train_result_raw_df         = pd.concat([list_train_raw_df[in_n], train_output_predict_raw_df, list_train_raw_df[out_n]], axis=1)
        test_output_predict_raw_df  = pd.DataFrame(test_output_predict_raw, columns = list_predict_feature_names[out_n])
        test_result_raw_df          = pd.concat([list_test_raw_df[in_n], test_output_predict_raw_df, list_test_raw_df[out_n]], axis=1)

        train_output_predict_std_df = pd.DataFrame(train_output_predict_std, columns = list_predict_feature_names[out_n])
        train_result_std_df         = pd.concat([list_train_std_df[in_n], train_output_predict_std_df, list_train_std_df[out_n]], axis=1)
        test_output_predict_std_df  = pd.DataFrame(test_output_predict_std, columns = list_predict_feature_names[out_n])
        test_result_std_df          = pd.concat([list_test_std_df[in_n], test_output_predict_std_df, list_test_std_df[out_n]], axis=1)

        train_output_predict_stdtoraw_df = pd.DataFrame(train_output_predict_stdtoraw, columns = list_predict_feature_names[out_n])
        train_result_stdtoraw_df         = pd.concat([list_train_std_df[in_n], train_output_predict_stdtoraw_df, list_train_std_df[out_n]], axis=1)
        test_output_predict_stdtoraw_df  = pd.DataFrame(test_output_predict_stdtoraw, columns = list_predict_feature_names[out_n])
        test_result_stdtoraw_df          = pd.concat([list_test_std_df[in_n], test_output_predict_stdtoraw_df, list_test_std_df[out_n]], axis=1)

        train_model_std_mse         = sklearn.metrics.mean_squared_error(list_train_std[out_n], train_output_predict_std)
        train_model_std_rmse        = np.sqrt(train_model_std_mse)
        test_model_std_mse          = sklearn.metrics.mean_squared_error(list_test_std[out_n], test_output_predict_std)
        test_model_std_rmse         = np.sqrt(test_model_std_mse)

        train_model_raw_mse         = sklearn.metrics.mean_squared_error(list_train_raw[out_n], train_output_predict_raw)
        train_model_raw_rmse        = np.sqrt(train_model_raw_mse)
        test_model_raw_mse          = sklearn.metrics.mean_squared_error(list_test_raw[out_n], test_output_predict_raw)
        test_model_raw_rmse         = np.sqrt(test_model_raw_mse)

        results_raw_df      = pd.DataFrame([model_name, train_model_raw_mse, train_model_raw_rmse, test_model_raw_mse, test_model_raw_rmse, train_model_raw_score, test_model_raw_score]).T
        results_std_df      = pd.DataFrame([model_name, train_model_std_mse, train_model_std_rmse, test_model_std_mse, test_model_std_rmse, train_model_std_score, test_model_std_score]).T

        results_raw_df.columns = columns_results
        results_std_df.columns = columns_results


        allmodel_results_raw_df = pd.concat([allmodel_results_raw_df, results_raw_df])
        allmodel_results_std_df = pd.concat([allmodel_results_std_df, results_std_df])

        train_result_raw_df.to_csv(parent_path / 'results' / theme_name / 'traintest_raw'/ (str(model_name) + '_train_raw.csv'), index=False)
        test_result_raw_df.to_csv(parent_path / 'results' / theme_name / 'traintest_raw'/ (str(model_name) + '_test_raw.csv'), index=False)
        train_result_std_df.to_csv(parent_path / 'results' / theme_name / 'traintest_std'/ (str(model_name) + '_train_std.csv'), index=False)
        test_result_std_df.to_csv(parent_path / 'results' / theme_name / 'traintest_std'/ (str(model_name) + '_test_std.csv'))
        train_result_stdtoraw_df.to_csv(parent_path / 'results' / theme_name / 'traintest_stdtoraw'/ (str(model_name) +  '_train_stdtoraw.csv'), index=False)
        test_result_stdtoraw_df.to_csv(parent_path / 'results' / theme_name / 'traintest_stdtoraw'/ (str(model_name) +  '_test_stdtoraw.csv'), index=False)

        #print('make the scatter diagram of raw')
        save_scatter_diagram(model_raw, model_name + '_raw')
        save_scatter_diagram(model_std, model_name + '_std')

        if hasattr(model_std, 'get_params') == True:
            model_params        = model_std.get_params()
            params_df           = pd.DataFrame([model_std.get_params])

        if hasattr(model_raw, 'intercept_') == True &  hasattr(model_raw, 'coef_') == True:
            model_intercept_raw_df  = pd.DataFrame(model_raw.intercept_)
            model_coef_raw_df       = pd.DataFrame(model_raw.coef_)
            model_parameter_raw_df  = pd.concat([model_intercept_raw_df, model_coef_raw_df])
            model_parameter_raw_df.to_csv(parent_path / 'results' / theme_name / 'parameter_raw' / (str(model_name) + '_parameter.csv'), index=False)

        if hasattr(model_std, 'intercept_') == True &  hasattr(model_std, 'coef_') == True:
            model_intercept_std_df  = pd.DataFrame(model_std.intercept_)
            model_coef_std_df       = pd.DataFrame(model_std.coef_)
            model_parameter_std_df  = pd.concat([model_intercept_std_df, model_coef_std_df])
            model_parameter_std_df.to_csv(parent_path / 'results' / theme_name / 'parameter_std' / (str(model_name) + '_parameter.csv'), index=False)

        if hasattr(model_raw, 'tree_') == True:
            save_tree_topdf(model_raw, model_name)
            pass


        if hasattr(model_raw, 'feature_importances_') == True:
            importances = pd.Series(model_raw.feature_importances_)
            importances = np.array(importances)

            label       = list_feature_names[in_n]

            # initialization of plt
            plt.clf()
            plt.bar(label, importances)
            plt.xticks(rotation=90)
            plt.xticks(fontsize=8)
            plt.rcParams["font.size"] = 12
            plt.title("importance in the tree " + str(theme_name))
            plt.savefig(parent_path / 'results' / theme_name / 'importance' / (str(model_name) + '.png'), dpi = 240)
            #plt.close()


        if hasattr(model_raw, 'estimators_') == True:
            #if 'DecisionTreeRegressor' in str(type(model_raw.estimators_[0])):
            if 'MultiOutput DecisionTreeRegressor' in model_name:

                MultiOutput_DTR_estimators = len(model_raw.estimators_)

                for i in range(MultiOutput_DTR_estimators):

                    model_name_i = model_name + '_'  + list_feature_names[out_n][i].replace('/', '_')


                    # save_importance_features
                    save_importance_features(model_raw.estimators_[i], model_name_i)

                    # save_tree to pdf
                    save_tree_topdf(model_raw.estimators_[i], model_name_i)

            #if 'RandomForestRegressor' in str(type(model_raw)):
            if 'RandomForest' in model_name:
                for i in range(1):
                    if hasattr(model_raw.estimators_[i], 'tree_') == True:
                        model_name_i = model_name + '_tree-'+ str(i)

                        # save_importance_features
                        save_importance_features(model_raw.estimators_[i], model_name_i)

                        # save_tree to pdf
                        save_tree_topdf(model_raw.estimators_[i], model_name_i)



            '''
            if 'XGB' in model_name:
                for i in range(3):

                    model_name_ = model_name + '_tree-'+ str(i)
                    graph = xgb.to_graphviz(model_raw.estimators_[i], num_trees = i)

                    graph.render('results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'tree' + os.sep
                                    + model_name_ +str(i)+ '.png')

                if hasattr(model_raw.estimators_[i], 'get_booster'):
                    # refer from https://github.com/dmlc/xgboost/issues/1238
                    #print('(model_raw.estimators_[i].get_booster().feature_names')
                    #print((model_raw.estimators_[i].get_booster().feature_names))

                    ##print(type((model_raw.estimators_[i].get_booster)))
                    #for x in dir(model_raw.estimators_[i].get_booster):
                    #    #print(x)
                    #sys.exit()
                    pass
                else:
                    #print('get booster not found')
                    pass
            '''


        # call gridsearch_predict
        if is_gridsearch == True:
            gridsearch_predict(model_raw, model_std, model_name)


        if is_bayesian_opt == True:
            # refer https://qiita.com/shinmura0/items/2b54ab0117727ce007fd
            # refer https://qiita.com/marshi/items/51b82a7b990d51bd98cd
            print('start the bayesian optimaization')

            def function_for_bayesian(x):
                optimize_type = var_bayesian.get()

                if optimize_type == 0:
                    #max
                    #print('max')
                    #print(model_std.predict(x)[0][-1] * -1)
                    return model_std.predict(x)[0][-1] * -1
                elif optimize_type == 1:
                    #min
                    #print('min')
                    #print(model_std.predict(x)[0][-1])
                    return model_std.predict(x)[0][-1]
                elif optimize_type ==2 and t_bayesian_val.get != '':
                    target_std_value = lastoutput_sc_model.transform(t_bayesian_val.get())[0][-1]

                    #print('target')
                    return (model_std.predict(x)[0][-1] - target_std_value)**2
                else:
                    return model_std.predict(x)[0][-1]

            bounds = []

            for i in range(list_num[in_n]):
                b_max = list_std_max[in_n][i]
                b_min = list_std_min[in_n][i]
                print(b_max)
                print(b_min)

                bounds.append({'name': list_feature_names[in_n][i] , 'type': 'continuous', 'domain': (list_std_min[in_n][i],list_std_max[in_n][i])})

            chkprint(bounds)
            myBopt = GPyOpt.methods.BayesianOptimization(f=function_for_bayesian, domain=bounds)
            myBopt.run_optimization(max_iter=25)

            print('result of bayesian optimization')
            #print([myBopt.x_opt])
            #print([myBopt.fx_opt])
            #print(list_sc_model[in_n].inverse_transform(np.array([myBopt.x_opt])))
            #print(lastoutput_sc_model.inverse_transform(np.array([myBopt.fx_opt])))

            optimized_input_df = pd.DataFrame(list_sc_model[in_n].inverse_transform(np.array([myBopt.x_opt])), columns = list_feature_names[in_n])
            optimized_output_df = pd.DataFrame(lastoutput_sc_model.inverse_transform(np.array([myBopt.fx_opt])),columns = lastoutput_feature_names)

            print(model_name)
            print('input')
            print(optimized_input_df)
            print('output')
            print(optimized_output_df)
            model_name_df = pd.Series(model_name)

            optimized_result_df = pd.concat([model_name_df, optimized_input_df, optimized_output_df], axis =1)
            optimized_result_df.to_csv(parent_path / 'results' / theme_name / 'bayesian_opt' / (str(model_name)+ '_bayesian_result.csv'), index=False)
            allmodel_bayesian_opt_df = pd.concat([allmodel_bayesian_opt_df, optimized_result_df])
            # end of bayesian_optimization

        # end of save_regression
        return model_raw, model_std


    def save_contour(model_raw, model_name):
        combinations_list_all = list(itertools.combinations(range(input_num),2))

        rank_tograph = 4
        important_list = [i for i in important_index if i < rank_tograph]
        combinations_list_selected = list(itertools.combinations(important_list,2))

        print('importances')
        print(importances)
        print('combinations_list_selected')
        print(combinations_list_selected)

        cnt_combi = 0

        for combi_ in combinations_list_selected:
            for others_inputtype in ['min', 'mean', 'max']:
                if others_inputtype == 'min':
                    input_others = input_raw_min
                elif others_inputtype == 'mean':
                    input_others = input_raw_mean
                elif others_inputtype == 'max':
                    input_others = input_raw_max

                grid_num = 9

                x = np.linspace(input_raw_min[combi_[0]], input_raw_max[combi_[0]], grid_num)
                y = np.linspace(input_raw_min[combi_[1]], input_raw_max[combi_[1]], grid_num)
                X, Y = np.meshgrid(x, y)

                input_Z_ =  np.array([input_others for i in range(grid_num * grid_num)])
                input_Z  = copy.deepcopy(input_Z_)
                input_Z  = input_Z.reshape(-1,input_num).tolist()

                cnt_ = 0
                for j in range(grid_num):
                    for k in range(grid_num):
                        input_Z[cnt_][combi_[0]] = X[j][k]
                        input_Z[cnt_][combi_[1]] = Y[j][k]
                        cnt_ +=1

                Z_all = (model_raw.predict(input_Z))

                for i in range(output_num):
                    Z_all = np.array(Z_all).reshape(grid_num * grid_num, -1)

                    Z = [k[i] for k in Z_all]
                    Z = np.array(Z).reshape(grid_num,-1).tolist()

                    X_name=input_feature_names[combi_[0]]
                    Y_name=input_feature_names[combi_[1]]
                    Z_name=output_feature_names[i]

                    plt_con = plt.figure()

                    cont = plt.contour(X,Y,Z,colors=['r', 'g', 'b'])
                    cont.clabel(fmt='%1.1f', fontsize=14)

                    plt.xlabel(X_name, fontsize=14)
                    plt.ylabel(Y_name, fontsize=14)

                    plt.pcolormesh(X,Y,Z, cmap='cool') #カラー等高線図
                    pp=plt.colorbar (orientation="vertical") # カラーバーの表示
                    pp.set_label(Z_name,  fontsize=24)
                    #plt.show()

                    plt.savefig(parent_path / 'results' / theme_name / 'predict_sr_contour' /
                                (str(Z_name) + '_'+ str(X_name) +'_' +  str(Y_name) +'_' +  others_inputtype +  '.png'))

                    if cnt_combi == 0 and others_inputtype == 'mean':
                        # contour

                        Image.MAX_IMAGE_PIXELS = None
                        img6 = Image.open(parent_path / 'results' / theme_name / 'predict_sr_contour' /
                                (str(Z_name) + '_'+ str(X_name) +'_' +  str(Y_name) +'_' +  others_inputtype +  '.png'))

                        img6_resize = img6.resize((photo_size, photo_size), Image.LANCZOS)
                        img6_resize.save(parent_path / 'results' / theme_name / 'predict_sr_contour' /
                                (str(Z_name) + '_'+ str(X_name) +'_' +  str(Y_name) +'_' +  others_inputtype +  '_resized.png'))


                        global image_contour
                        image_open = Image.open(parent_path / 'results' / theme_name / 'predict_sr_contour' /
                                (str(Z_name) + '_'+ str(X_name) +'_' +  str(Y_name) +'_' +  others_inputtype +  '_resized.png'))
                        image_contour = ImageTk.PhotoImage(image_open, master=frame2)

                        canvas_contour.create_image(int(photo_size/2),int(photo_size/2), image=image_contour)
                        #plt.close()


                    plt.close(plt_con)

            cnt_combi += 1


    def save_summary(model_raw, model_std, model_name):
        # save the RMSE, MSE, R2 of model (of major model)

        global summary_results_raw_df
        global summary_results_std_df

        train_output_predict_raw    = model_raw.predict(list_train_raw[in_n])
        test_output_predict_raw     = model_raw.predict(list_test_raw[in_n])
        train_output_predict_std    = model_std.predict(list_train_std[in_n])
        test_output_predict_std     = model_std.predict(list_test_std[in_n])

        train_output_predict_stdtoraw    = model_std.predict(list_train_std[in_n])
        test_output_predict_stdtoraw     = model_std.predict(list_test_std[in_n])
        train_output_predict_stdtoraw    = list_sc_model[out_n].inverse_transform(train_output_predict_std)
        test_output_predict_stdtoraw     = list_sc_model[out_n].inverse_transform(test_output_predict_std)

        if hasattr(model_raw, 'score') == True:
            train_model_raw_score   = model_raw.score(list_train_raw[in_n] , list_train_raw[out_n])
            train_model_raw_score   = np.round(train_model_raw_score, 5)
            test_model_raw_score    = model_raw.score(list_test_raw[in_n],   list_test_raw[out_n])
            test_model_raw_score    = np.round(test_model_raw_score, 5)

        if hasattr(model_raw, 'evaluate') == True:
            train_model_raw_score   = model_raw.evaluate(list_train_raw[in_n] , list_train_raw[out_n])
            train_model_raw_score   = np.round(train_model_raw_score, 5)
            test_model_raw_score    = model_raw.evaluate(test_input_raw,   list_test_raw[out_n])
            test_model_raw_score    = np.round(test_model_raw_score, 5)

        if hasattr(model_std, 'score') == True:
            train_model_std_score   = model_std.score(list_train_std[in_n] , list_train_std[out_n])
            train_model_std_score   = np.round(train_model_std_score, 5)
            test_model_std_score    = model_std.score(list_test_std[in_n],   list_test_std[out_n])
            test_model_std_score    = np.round(test_model_std_score, 5)

        if hasattr(model_std, 'evaluate') == True:
            train_model_std_score   = model_std.evaluate(list_train_std[in_n] , list_train_std[out_n])
            train_model_std_score   = np.round(train_model_std_score, 5)
            test_model_std_score    = model_std.evaluate(test_input_std,   list_test_std[out_n])
            test_model_std_score    = np.round(test_model_std_score, 5)

        train_output_predict_raw_df = pd.DataFrame(train_output_predict_raw, columns = list_predict_feature_names[out_n])
        train_result_raw_df         = pd.concat([list_train_raw_df[in_n], train_output_predict_raw_df, list_train_raw_df[out_n]], axis=1)
        test_output_predict_raw_df  = pd.DataFrame(test_output_predict_raw, columns = list_predict_feature_names[out_n])
        test_result_raw_df          = pd.concat([list_test_raw_df[in_n], test_output_predict_raw_df, list_test_raw_df[out_n]], axis=1)

        train_output_predict_std_df = pd.DataFrame(train_output_predict_std, columns = list_predict_feature_names[out_n])
        train_result_std_df         = pd.concat([list_train_std_df[in_n], train_output_predict_std_df, list_train_std_df[out_n]], axis=1)
        test_output_predict_std_df  = pd.DataFrame(test_output_predict_std, columns = list_predict_feature_names[out_n])
        test_result_std_df          = pd.concat([list_test_std_df[in_n], test_output_predict_std_df, list_test_std_df[out_n]], axis=1)

        train_output_predict_stdtoraw_df = pd.DataFrame(train_output_predict_stdtoraw, columns = list_predict_feature_names[out_n])
        train_result_stdtoraw_df         = pd.concat([list_train_std_df[in_n], train_output_predict_stdtoraw_df, list_train_std_df[out_n]], axis=1)
        test_output_predict_stdtoraw_df  = pd.DataFrame(test_output_predict_stdtoraw, columns = list_predict_feature_names[out_n])
        test_result_stdtoraw_df          = pd.concat([list_test_std_df[in_n], test_output_predict_stdtoraw_df, list_test_std_df[out_n]], axis=1)

        train_model_std_mse         = sklearn.metrics.mean_squared_error(list_train_std[out_n], train_output_predict_std)
        train_model_std_rmse        = np.sqrt(train_model_std_mse)
        test_model_std_mse          = sklearn.metrics.mean_squared_error(list_test_std[out_n], test_output_predict_std)
        test_model_std_rmse         = np.sqrt(test_model_std_mse)

        train_model_raw_mse         = sklearn.metrics.mean_squared_error(list_train_raw[out_n], train_output_predict_raw)
        train_model_raw_rmse        = np.sqrt(train_model_raw_mse)
        test_model_raw_mse          = sklearn.metrics.mean_squared_error(list_test_raw[out_n], test_output_predict_raw)
        test_model_raw_rmse         = np.sqrt(test_model_raw_mse)

        results_raw_df      = pd.DataFrame([model_name, train_model_raw_mse, train_model_raw_rmse, test_model_raw_mse, test_model_raw_rmse, train_model_raw_score, test_model_raw_score]).T
        results_std_df      = pd.DataFrame([model_name, train_model_std_mse, train_model_std_rmse, test_model_std_mse, test_model_std_rmse, train_model_std_score, test_model_std_score]).T

        results_raw_df.columns = columns_results
        results_std_df.columns = columns_results

        summary_results_raw_df = pd.concat([summary_results_raw_df, results_raw_df])
        summary_results_std_df = pd.concat([summary_results_std_df, results_std_df])


    if output_num != 1:
        list_inverse_predict = [False]
    elif output_num == 1:
        list_inverse_predict = [False]


    for is_inverse_predict in list_inverse_predict: # 0,1
        chkprint(is_inverse_predict)

        # False  0: forward predict *normal
        # True   1: inverse predict

        inv_    = is_inverse_predict
        in_n    = is_inverse_predict        # 0 to 1
        out_n   = not(is_inverse_predict)   # 1 to 0

        direction_name_list = ['normal', 'inverse']
        direction_name      = direction_name_list[inv_]

        columns_results = ['model_name',
                            'train_model_mse',
                            'train_model_rmse',
                            'test_model_mse',
                            'test_model_rmse',
                            'train_model_score',
                            'test_model_score']

        #allmodel_results_raw_df = pd.DataFrame(columns = columns_results)
        #allmodel_results_std_df = pd.DataFrame(columns = columns_results)
        #allmodel_bayesian_opt_df = pd.DataFrame(columns = list_feature_names[in_n])


        #########   predict of all candidate by the scikitlearn model ###############

        # select the feature value by the random forest regressor
        max_depth = 7
        model       = sklearn.ensemble.RandomForestRegressor(max_depth = max_depth)
        model_name  = ''
        model_name  += 'RandomForestRegressor_'
        model_name  += 'max_depth_'+str(max_depth)

        model.fit(list_train_raw[in_n], list_train_raw[out_n])

        importances     = np.array(model.feature_importances_)
        label           = list_feature_names[in_n]
        #important_rank = pd.DataFrame(importances).rank(method="first", ascending=False)
        #print('important_rank', important_rank)

        important_index = np.array(-importances).argsort()

        print('important_index', important_index)

        pred_train = model.predict(list_train_raw[in_n])
        pred_test = model.predict(list_test_raw[in_n])

        from PIL import Image

        plt.figure(figsize=(5,5))
        plt.scatter(list_train_raw[out_n], pred_train, label = 'Train', c = 'blue')
        plt.title(model_name)
        plt.xlabel('Measured value')
        plt.ylabel('Predicted value')
        plt.scatter(list_test_raw[out_n], pred_test, c = 'lightgreen', label = 'Test', alpha = 0.8)
        plt.legend(loc = 4)
        plt.savefig(parent_path / 'results' / theme_name / 'meas_pred.png')
        #plt.close()

        img1 = Image.open(parent_path / 'results' / theme_name / 'meas_pred.png')

        img1_resize = img1.resize((photo_size, photo_size), Image.LANCZOS)
        img1_resize.save(parent_path / 'results' / theme_name / 'meas_pred.png')

        global image_predicted_values
        image_open = Image.open(parent_path / 'results' / theme_name / 'meas_pred.png')
        image_predicted_values = ImageTk.PhotoImage(image_open, master=frame2)

        canvas_predicted_values.create_image(int(photo_size/2),int(photo_size/2), image=image_predicted_values)


        ########################
        plt.figure(figsize =(5,5))
        plt.bar(label, importances)

        plt.xticks(rotation=90)
        plt.xticks(fontsize=8)
        plt.rcParams["font.size"] = 12

        plt.title("-" + model_name)
        plt.savefig(parent_path / 'results' / theme_name / 'tmp_importances.png', dpi = 240)
        #plt.close()

        img2 = Image.open(parent_path / 'results' / theme_name / 'tmp_importances.png')

        img2_resize = img2.resize((photo_size, photo_size), Image.LANCZOS)
        img2_resize.save(parent_path / 'results' / theme_name / 'tmp_importances.png')

        global image_important_variable
        image_open = Image.open(parent_path / 'results' / theme_name / 'tmp_importances.png')
        image_important_variable = ImageTk.PhotoImage(image_open, master=frame2)

        canvas_important_variable.create_image(int(photo_size/2),int(photo_size/2), image=image_important_variable)



        global image_correlation_coefficient

        img3 = Image.open(parent_path / 'results' / theme_name / 'correlation_coefficient.png')
        img3_resize = img3.resize((photo_size, photo_size), Image.LANCZOS)
        img3_resize.save(parent_path / 'results' / theme_name / 'correlation_coefficient.png')
        image_open = Image.open(parent_path / 'results' / theme_name / 'correlation_coefficient.png')

        image_correlation_coefficient = ImageTk.PhotoImage(image_open, master=frame2)

        canvas_correlation_coefficient.create_image(int(photo_size/2),int(photo_size/2), image=image_correlation_coefficient)


        chkprint(importances)
        importances_sort    = importances.argsort()[::-1]
        split_base          = np.array([15,13,9,4,4,3,3,3]) # max:758160
        split_base          = np.array([10,7,3,3,3,3,3,3])  # max:51030

        # set the split num from importances rank of random forest regressor
        split_num   = np.full(len(importances_sort),1)
        for i in range(min(len(importances),8)):
            rank_ = importances_sort[i]
            split_num[rank_] = split_base[i]

        def combination(max, min, split_num):
            candidate = []
            for i in range(list_num[in_n]):
                candidate.append(np.linspace(start = max[i], stop = min[i], num = split_num[i]))

            candidate = np.array(candidate)
            return candidate

        if is_gridsearch == True:
            all_gridsearch_number = split_num.prod()
            candidate = combination(list_raw_max[in_n], list_raw_min[in_n], split_num)

            # refer from https://teratail.com/questions/152110
            # unpack   *candidate
            gridsearch_input_raw    = list(itertools.product(*candidate))
            #print(gridsearch_input_raw)
            gridsearch_input_std    = list_sc_model[in_n].transform(gridsearch_input_raw)

            gridsearch_input_raw_df = pd.DataFrame(gridsearch_input_raw, columns = list_feature_names[in_n])
            gridsearch_input_std_df = pd.DataFrame(gridsearch_input_std, columns = list_feature_names[in_n])


        n_trials= 2 + Booleanvar_optuna_sklearn.get()*77


        ##################### Linear Regression #####################

        model = linear_model.LinearRegression()
        model_name = 'Linear_Regression_'

        model_raw, model_std = fit_model_std_raw(model, model_name)
        model_name = 'Linear'
        save_summary(model_raw, model_std, model_name)

        ##################### heil–Sen #####################

        model = MultiOutputRegressor(linear_model.TheilSenRegressor())
        model_name = 'TheilSen_'

        model_raw, model_std = fit_model_std_raw(model, model_name)
        LinearRegression_model = model
        LinearRegression_model_name = model_name
        model_name = 'TheilSen'

        save_summary(model_raw, model_std, model_name)

        ##################### Regression of Stochastic Gradient Descent #####################
        max_iter = 1000

        model = MultiOutputRegressor(linear_model.SGDRegressor(max_iter = max_iter))
        model_name = 'MO_SGD_'
        model_name += 'max_i_'+str(max_iter)

        model_raw, model_std = fit_model_std_raw(model, model_name)
        Multi_SGD_model = model
        Multi_SGD_model_name = model_name
        model_name = 'MO-SGD'
        save_summary(model_raw, model_std, model_name)
        ##################### Regression of SVR #####################

        kernel_ = ['rbf']
        C_= [0.1, 1, 100]

        for kernel_, C_ in itertools.product(kernel_, C_):
            model = MultiOutputRegressor(svm.SVR(kernel = kernel_, C = C_))
            model_name = 'MO_SVR_'
            model_name += '_k_'+str(kernel_)
            model_name += '_C_'+str(C_)

            model_raw, model_std = fit_model_std_raw(model, model_name)


        def objective_svr(trial):
            svr_c = trial.suggest_loguniform('svr_c', 1e-2, 1e6)
            epsilon = trial.suggest_loguniform('epsilon', 1e-7, 1e7)
            svr = MultiOutputRegressor(svm.SVR(kernel = 'rbf', C = svr_c, epsilon=epsilon))
            svr.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = svr.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_svr, n_trials=n_trials)

        for kernel_, C_, epsilon in [('rbf', study.best_params['svr_c'], study.best_params['epsilon'])]:
            model = MultiOutputRegressor(svm.SVR(kernel = kernel_, C = C_, epsilon = epsilon))
            model_name = 'MO_SVR_best_'
            model_name += '_k_'+str(kernel_)
            model_name += '_C_'+str(np.round(C_,2))
            model_name += '_e_'+str(np.round(epsilon,2))

            model_raw, model_std = fit_model_std_raw(model, model_name)

            model_name = 'SVR'
            save_summary(model_raw, model_std, model_name)


        # refer https://www.slideshare.net/ShinyaShimizu/ss-11623505

        ##################### Regression of Ridge #####################
        for alpha in [0.01, 0.1, 1.0] :
            model = linear_model.Ridge(alpha = alpha)
            model_name = 'Ridge_'
            model_name += 'a_'+str(alpha)

            model_raw, model_std = fit_model_std_raw(model, model_name)

        def objective_ridge(trial):
            alpha = trial.suggest_loguniform('alpha', 1e0, 1e6)

            model = linear_model.Ridge(alpha = alpha)
            model.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = model.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_ridge, n_trials=n_trials)

        for alpha in [(study.best_params['alpha'])]:
            model = linear_model.Ridge(alpha = alpha)
            model_name = 'Ridge_best_'
            model_name += 'a_'+str(np.round(alpha,2))

            model_raw, model_std = fit_model_std_raw(model, model_name)
            model_name = 'Ridge'
            save_summary(model_raw, model_std, model_name)

        ##################### Regression of KernelRidge #####################
        for alpha in [0.01, 1.0 ,100]:
            model = KernelRidge(alpha=alpha, kernel='rbf')
            model_name = 'KRidge_'
            model_name += 'a_'+str(alpha)

            model_raw, model_std = fit_model_std_raw(model, model_name)


        def objective_kridge(trial):
            alpha = trial.suggest_loguniform('alpha', 1e0, 1e4)

            model = KernelRidge(alpha=alpha, kernel='rbf')
            model.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = model.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_kridge, n_trials=n_trials)

        for alpha in [(study.best_params['alpha'])]:
            model = linear_model.Ridge(alpha = alpha)
            model_name = 'KRidge_best_'
            model_name += 'a_'+str(np.round(alpha,2))

            model_raw, model_std = fit_model_std_raw(model, model_name)
            model_name = 'KRidge'
            save_summary(model_raw, model_std, model_name)


        ##################### Regression of Lasso #####################
        for alpha in [0.01, 0.1, 1.0]:
            model = linear_model.Lasso(alpha = alpha)
            model_name = 'Lasso_'
            model_name += 'a_'+str(alpha)

            model_raw, model_std = fit_model_std_raw(model, model_name)

        def objective_lasso(trial):
            alpha = trial.suggest_loguniform('alpha', 1e0, 1e4)

            model = linear_model.Lasso(alpha = alpha)
            model.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = model.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_lasso, n_trials=n_trials)

        for alpha in [(study.best_params['alpha'])]:
            model = linear_model.Lasso(alpha = alpha)
            model_name = 'Lasso_best_'
            model_name += 'a_'+str(np.round(alpha,2))

            model_raw, model_std = fit_model_std_raw(model, model_name)
            model_name = 'Lasso'
            save_summary(model_raw, model_std, model_name)


        ##################### Regression of Elastic Net #####################

        alpha      = [0.01, 0.1]
        l1_ratio   = [0.25, 0.75]

        for alpha, l1_ratio in itertools.product(alpha, l1_ratio):

            model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            model_name = 'ENet_'
            model_name += 'a_'+str(alpha)
            model_name += 'l1_r_'+str(l1_ratio)

            model_raw, model_std = fit_model_std_raw(model, model_name)

        def objective_enet(trial):
            alpha = trial.suggest_loguniform('alpha', 1e0, 1e4)
            l1_ratio = trial.suggest_uniform('l1_ratio', 0, 1)

            model = linear_model.ElasticNet(alpha=alpha, l1_ratio = l1_ratio)
            model.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = model.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_enet, n_trials=n_trials)


        for alpha, l1_ratio in [(study.best_params['alpha'], study.best_params['l1_ratio'])]:
            model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            model_name = 'ENet_best_'
            model_name += 'a_'+str(np.round(alpha,2))
            model_name += 'l1_r_'+str(np.round(l1_ratio,2))

            model_raw, model_std = fit_model_std_raw(model, model_name)
            model_name = 'EN'
            save_summary(model_raw, model_std, model_name)


        ##################### Regression of MultiTaskLassoCV #####################
        max_iter_ = 1000

        model = linear_model.MultiTaskLassoCV()
        model_name = 'MT-Lasso_'
        model_name += 'max_i_'+str(max_iter)

        model_raw, model_std = fit_model_std_raw(model, model_name)

        ##################### Regression of Multi Task Elastic Net CV #####################
        model = linear_model.MultiTaskElasticNetCV()

        model_name = 'MT-ENet_'
        model_raw, model_std = fit_model_std_raw(model, model_name)

        ##################### Regression of OrthogonalMatchingPursuit #####################
        #model = linear_model.OrthogonalMatchingPursuit()
        #model_name = 'OrthogonalMatchingPursuit_'

        #model_raw, model_std = fit_model_std_raw(model, model_name)

        ##################### Regression of BayesianRidge #####################
        model = MultiOutputRegressor(linear_model.BayesianRidge())
        model_name = 'MO_BRidge_'

        model_raw, model_std = fit_model_std_raw(model, model_name)


        def objective_bridge(trial):
            alpha_1 = trial.suggest_loguniform('alpha_1', 1e-8, 1e-4)
            alpha_2 = trial.suggest_loguniform('alpha_2', 1e-8, 1e-4)
            lambda_1 = trial.suggest_loguniform('lambda_1', 1e-8, 1e-4)
            lambda_2 = trial.suggest_loguniform('lambda_2', 1e-8, 1e-4)

            model = MultiOutputRegressor(
                        linear_model.BayesianRidge(
                                        alpha_1=alpha_1,
                                        alpha_2=alpha_2,
                                        lambda_1=lambda_1,
                                        lambda_2=lambda_2,
                                        ))
            model.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = model.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_bridge, n_trials=n_trials)

        for alpha_1, alpha_2, lambda_1, lambda_2 in [(
                                                              study.best_params['alpha_1'],
                                                              study.best_params['alpha_2'],
                                                              study.best_params['lambda_1'],
                                                              study.best_params['lambda_2']
                                                              )]:
            model = MultiOutputRegressor(
                                        linear_model.BayesianRidge(
                                            n_iter=300,
                                            tol=0.001,
                                            alpha_1=alpha_1,
                                            alpha_2=alpha_2,
                                            lambda_1=lambda_1,
                                            lambda_2=lambda_2
                                            ))
            model_name = 'MO_BRidge_best'
            model_name += '_a1_'+str(np.round(np.log(alpha_1),1))
            model_name += '_a2_'+str(np.round(np.log(alpha_2),1))
            model_name += '_l1_'+str(np.round(np.log(lambda_1),1))
            model_name += '_l2_'+str(np.round(np.log(lambda_2),1))
            model_raw, model_std = fit_model_std_raw(model, model_name)
            model_name = 'BRidge'
            save_summary(model_raw, model_std, model_name)



        ##################### Regression of GaussianProcessRegressor #####################
        from sklearn.gaussian_process import GaussianProcessRegressor

        model = MultiOutputRegressor(GaussianProcessRegressor())
        model_name = 'MO-GPR_'

        model_raw, model_std = fit_model_std_raw(model, model_name)

        def objective_gpr(trial):
            alpha = trial.suggest_loguniform('alpha', 1e-14, 1e-6)

            model = MultiOutputRegressor(GaussianProcessRegressor(alpha=alpha))
            model.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = model.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_gpr, n_trials=n_trials)

        # 最適解
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)

        for alpha in           [(
                                study.best_params['alpha']
                                )]:
            model = MultiOutputRegressor(GaussianProcessRegressor(alpha=alpha))
            model_name = 'MO-GPR_best'
            model_name += '_a_'+str(np.round(np.log(alpha),2))
            model_raw, model_std = fit_model_std_raw(model, model_name)
            model_name = 'GPR'
            save_summary(model_raw, model_std, model_name)


        ##################### Regression of DecisionTreeRegressor #####################

        for max_depth in [7,10]:
            model = sklearn.tree.DecisionTreeRegressor(max_depth = max_depth)
            model_name = 'Tree_'
            model_name += 'max_d_'+str(max_depth)

            model_raw, model_std = fit_model_std_raw(model, model_name)


        def objective_dtr(trial):
            max_depth = trial.suggest_int('max_depth', 2, 13)

            model = sklearn.tree.DecisionTreeRegressor(max_depth = max_depth)
            model.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = model.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_dtr, n_trials=n_trials)

        # 最適解
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)

        for alpha in           [(
                                study.best_params['max_depth']
                                )]:
            model = sklearn.tree.DecisionTreeRegressor(max_depth = max_depth)
            model_name = 'Tree_best_'
            model_name += 'max_d_'+str(max_depth)
            model_raw, model_std = fit_model_std_raw(model, model_name)
            model_name = 'DTR'
            save_summary(model_raw, model_std, model_name)





        ##################### Regression of Multioutput DecisionTreeRegressor #####################
        tmp_r2_score = 0
        for max_depth in [3,5,9]:

            model = MultiOutputRegressor(sklearn.tree.DecisionTreeRegressor(max_depth = max_depth))
            model_name = 'MO_Tree_'
            model_name += 'max_d_' + str(max_depth)
            model_raw, model_std = fit_model_std_raw(model, model_name)

        def objective_modtr(trial):
            max_depth = trial.suggest_int('max_depth', 2, 13)

            model = MultiOutputRegressor(sklearn.tree.DecisionTreeRegressor(max_depth = max_depth))
            model.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = model.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_modtr, n_trials=n_trials)

        # 最適解
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)


        for alpha in           [(
                                study.best_params['max_depth']
                                )]:
            model = MultiOutputRegressor(sklearn.tree.DecisionTreeRegressor(max_depth = max_depth))
            model_name = 'MO_Tree_best_'
            model_name += 'max_d_'+str(max_depth)
            model_raw, model_std = fit_model_std_raw(model, model_name)
            model_name = 'M-DTR'
            save_summary(model_raw, model_std, model_name)


        #################### Regression of RandomForestRegressor #####################
        for max_depth in [3,5,7,9,11]:
            model = sklearn.ensemble.RandomForestRegressor(max_depth = max_depth)
            model_name = ''
            model_name += 'RandForest_'
            #model_name += get_variablename(max_depth)
            model_name += 'max_d_'+str(max_depth)

            model_raw, model_std = fit_model_std_raw(model, model_name)


        def objective_rfr(trial):
            max_depth = trial.suggest_int('max_depth', 2, 13)

            model = sklearn.ensemble.RandomForestRegressor(max_depth = max_depth)
            model.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = model.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_rfr, n_trials=n_trials)

        # 最適解
        print(study.best_params)
        print(study.best_value)
        print(study.best_trial)

        for alpha in           [(
                                study.best_params['max_depth']
                                )]:
            model = sklearn.ensemble.RandomForestRegressor(max_depth = max_depth)
            model_name = 'RandForest_best_'
            model_name += 'max_d_'+str(max_depth)
            model_raw, model_std = fit_model_std_raw(model, model_name)
            model_name = 'RFR'
            save_summary(model_raw, model_std, model_name)



        ##################### Regression of XGBoost #####################
        # refer from https://github.com/FelixNeutatz/ED2/blob/23170b05c7c800e2d2e2cf80d62703ee540d2bcb/src/model/ml/CellPredict.py

        min_child_weight = [5] #1,3
        subsample        = [0.9] #0.7, 0.8,
        learning_rate    = [0.1,0.01] #0.1
        max_depth        = [7]
        n_estimators      = [100]

        tmp_r2_score = 0
        for min_child_weight, subsample, learning_rate, max_depth, n_estimators \
            in itertools.product(min_child_weight, subsample, learning_rate, max_depth, n_estimators):

            xgb_params = {'estimator__min_child_weight': min_child_weight,
                          'estimator__subsample': subsample,
                          'estimator__learning_rate': learning_rate,
                          'estimator__max_depth': max_depth,
                          'estimator__n_estimators': n_estimators,
                          'colsample_bytree': 0.8,
                          'silent': 1,
                          'seed': 0,
                          'objective': 'reg:linear'}

            model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))

            model_name = 'MO-XGB'
            model_name += 'm_c_w_'+str(min_child_weight)
            model_name += 'ss_'+str(subsample)
            model_name += 'l_r_'+str(learning_rate)
            model_name += 'm_d_'+str(max_depth)
            model_name += 'n_es_'+str(n_estimators)

            model_raw, model_std = fit_model_std_raw(model, model_name)

        def objective_xgb(trial):

            min_child_weight = trial.suggest_int('min_child_weight', 1 , 10)
            subsample        = trial.suggest_uniform('subsample', 0.1, 1.0)
            learning_rate    = trial.suggest_loguniform('learning_rate', 1e-2, 1e+1)
            max_depth        = trial.suggest_int('max_depth', 3 , 10)
            n_estimators     = 100

            xgb_params = {'estimator__min_child_weight': min_child_weight,
                          'estimator__subsample': subsample,
                          'estimator__learning_rate': learning_rate,
                          'estimator__max_depth': max_depth,
                          'estimator__n_estimators': n_estimators,
                          'colsample_bytree': 0.8,
                          'silent': 1,
                          'seed': 0,
                          'objective': 'reg:linear'}


            model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))
            model.fit(list_train_raw[in_n], list_train_raw[out_n])
            y_pred = model.predict(list_val_raw[in_n])

            return mean_squared_error(list_val_raw[out_n], y_pred)

        study = optuna.create_study()
        study.optimize(objective_xgb, n_trials=n_trials)

        for min_child_weight, subsample, learning_rate, max_depth in \
                                [(
                                study.best_params['min_child_weight'],
                                study.best_params['subsample'],
                                study.best_params['learning_rate'],
                                study.best_params['max_depth']
                                )]:

            xgb_params = {'estimator__min_child_weight': min_child_weight,
                          'estimator__subsample': subsample,
                          'estimator__learning_rate': learning_rate,
                          'estimator__max_depth': max_depth,
                          'estimator__n_estimators': 100,
                          'colsample_bytree': 0.8,
                          'silent': 1,
                          'seed': 0,
                          'objective': 'reg:linear'}

            model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))

            model_name =  'MO-XGB_best'
            model_name += '_c_wei_'+str(min_child_weight)
            model_name += '_sam_'+str(np.round(subsample,1))
            model_name += '_rate_'+str(np.round(np.log(learning_rate),1))
            model_name += '_dep_'+str(max_depth)
            model_name += '_n_es_'+str(n_estimators)
            model_raw, model_std = fit_model_std_raw(model, model_name)
            model_name = 'XGB'
            save_summary(model_raw, model_std, model_name)

            save_contour(model_raw, model_name)


        ################# to csv ##############################
        allmodel_results_raw_df.to_csv(os.path.join(parent_path, 'results', theme_name, 'comparison of all_methods_raw.csv'), index=False)
        allmodel_results_std_df.to_csv(os.path.join(parent_path, 'results', theme_name, 'comparison of all_methods_std.csv'), index=False)

        summary_results_raw_df.to_csv(os.path.join(parent_path, 'results', theme_name, 'summary of methods_raw.csv'), index=False)
        summary_results_std_df.to_csv(os.path.join(parent_path, 'results', theme_name, 'summary of methods_std.csv'), index=False)

        ################# to photo ##############################
        # R2 score
        plt.figure()
        #https://own-search-and-study.xyz/2016/08/03/pandas%E3%81%AEplot%E3%81%AE%E5%85%A8%E5%BC%95%E6%95%B0%E3%82%92%E4%BD%BF%E3%81%84%E3%81%93%E3%81%AA%E3%81%99/
        summary_results_std_df.plot(kind='bar', x='model_name', y =['train_model_score', 'test_model_score'], rot=70, figsize=(12,12), fontsize=18, yticks=[0,0.5,1.0], ylim=[0,1.0])
        plt.savefig(parent_path / 'results' /  theme_name / 'summary_of_score.png', dpi = 240)

        Image.MAX_IMAGE_PIXELS = None
        img6 = Image.open(parent_path / 'results' / theme_name / 'summary_of_score.png')
        img6_resize = img6.resize((photo_size, photo_size), Image.LANCZOS)
        img6_resize.save(parent_path / 'results' / theme_name / 'summary_of_score_resized.png')

        global image_score
        image_open = Image.open(parent_path / 'results' / theme_name / 'summary_of_score_resized.png')
        image_score = ImageTk.PhotoImage(image_open, master=frame2)

        canvas_score.create_image(int(photo_size/2),int(photo_size/2), image=image_score)

        if t_bayesian_val.get().isnumeric == True:
            target_std_value = list_sc_model[out_n].transform(t_bayesian_val.get())
        else:
            target_std_value = 'None'


        optimize_dic = {0: 'max', 1: 'min', 2:target_std_value}

        optimize_type = var_bayesian.get()
        print(optimize_dic[optimize_type])

        allmodel_bayesian_opt_df.to_csv(os.path.join(parent_path, 'results', theme_name, 'bayesian_opt_' +str(optimize_dic[optimize_type])+  '.csv'), index=False)
        #######################################################

        print('Sklearn finished!')

        '''
        ################# importances feature by XGBOOST ######
        import matplotlib.pyplot as plt

        importances = pd.Series(reg1_multigbtree.feature_importances_)
        importances = importances.sort_values()
        importances.plot(kind = "barh")
        plt.title("imporance in the xgboost Model")
        plt.show()
        #######################################################
        '''

        '''
        ##################### LIME Explainer #####################
        import lime
        import lime.lime_tabular

        #explainer1 = lime.lime_tabular.LimeTabularExplainer(train_output, feature_names=input_feature_names, kernel_width=3)
        0
        explainer1 = lime.lime_tabular.LimeTabularExplainer(train_input, feature_names= input_feature_names, class_names=output_feature_names, verbose=True, mode='regression')

        np.random.seed(1)
        i = 3
        #exp = explainer.explain_instance(test[2], predict_fn, num_features=10)
        exp = explainer.explain_instance(test[i], reg1_SVR.predict, num_features=5)

        sys.exit()
        # exp.show_in_notebook(show_all=False)
        exp.save_to_file(file_path= str(address_) + 'numeric_category_feat_01', show_table=True, show_all=True)

        i = 3
        exp = explainer.explain_instance(test[i], predict_fn, num_features=10)
        # exp.show_in_notebook(show_all=False)
        exp.save_to_file(file_path=str(address_) + 'numeric_category_feat_02', show_table=True, show_all=True)
        ##########################################################
        '''

        '''
        # import pickle
        # pickle.dump(reg, open("model.pkl", "wb"))
        # reg = pickle.load(open("model.pkl", "rb"))

        pred1_train = reg1_gbtree.predict(train_input)
        pred1_test = reg1_gbtree.predict(test_input)
        #print(mean_squared_error(train_output, pred1_train))
        #print(mean_squared_error(test_output, pred1_test))

        import matplotlib.pyplot as plt

        importances = pd.Series(reg1_gbtree.feature_importances_)
        importances = importances.sort_values()
        importances.plot(kind = "barh")
        plt.title("imporance in the xgboost Model")
        plt.show()
        '''

    ##################### Deep Learning #####################
    if is_dl == False:
        print('You didn\'t choose deeplearning')
        print('Finished!')

    elif is_dl == True :
        print('start deeplearning')

        import keras
        from keras import backend as K
        import keras.models

        from keras.models import Sequential, load_model
        from keras.layers import Activation, InputLayer, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
        from keras.layers.normalization import BatchNormalization
        from keras.callbacks import EarlyStopping
        from keras.wrappers.scikit_learn import KerasRegressor
        from keras.utils import plot_model

        import h5py


        def get_model(layers_depth, units_size, keep_prob, patience):

            layers_depth = int(float(layers_depth))
            units_size = int(float(units_size))

            print('layers_depth :',layers_depth)
            print('units_size :',units_size)
            print('keep_prob :',keep_prob)
            print('patience :',patience)

            model = Sequential()
            model.add(Dense(units_size, input_shape=(input_num,)))
            model.add(Activation('relu'))
            model.add(BatchNormalization(mode=0))

            for i in range(layers_depth-2):
                model.add(Dense(units_size))
                model.add(Activation('relu'))
                model.add(BatchNormalization(mode=0))
            #model.add(Dense(1))
            model.add(Dense(output_num))


            model.compile(loss='mse',
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])

            return model

        # refer from https://github.com/completelyAbsorbed/ML/blob/0ca17d25bae327fe9be8e3639426dc86f3555a5a/Practice/housing/housing_regression_NN.py


        layers_depth  = [4,5]
        units_size  = [64, 32, 16]
        #bn_where    = [3, 0]
        ac_last     = [0, 1]
        keep_prob   = [0]
        patience    = [3000]


        for dl_params in itertools.product(layers_depth, units_size, keep_prob, patience):
            layers_depth, units_size, keep_prob, patience = dl_params

            batch_size  = 30
            epochs   = 150
            cb = EarlyStopping(monitor='loss', patience=patience, mode='auto')

            model_raw = get_model(*dl_params)
            model_std = get_model(*dl_params)

            model_name =   'dl'
            model_name +=  '_depth-'     + str(layers_depth)
            model_name +=  '_unit-'      + str(units_size)
            model_name +=  '_drop-'      + str(keep_prob)
            model_name +=  '_patience-'  + str(patience)


            model_raw.fit(list_train_raw[in_n], list_train_raw[out_n],
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=[cb],
                      verbose=1)

            model_std.fit(list_train_std[in_n], list_train_std[out_n],
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=[cb],
                      verbose=1)

            save_regression(model_raw, model_std, model_name)


        def objective_dl(trial):
            def get_model_1(layers_depth, units_size, keep_prob, patience):
                return model

            layers_depth    = trial.suggest_int('layers_depth', 3 , 10)
            units_size      = trial.suggest_int('units_size', 10, 1000)
            keep_prob       = trial.suggest_uniform('keep_prob', 0, 0.9)
            patience        = trial.suggest_int('patience', 5 , 10000)

            dl_params = {'layers_depth': layers_depth,
                        'units_size': units_size,
                        'keep_prob': keep_prob,
                        'patience': patience
                        }


            model = get_model(**dl_params)
            epochs=50
            model.fit(list_train_std[in_n], list_train_std[out_n],
                    batch_size=batch_size,
                      epochs=epochs,
                      verbose=1)

            y_pred = model.predict(list_val_std[in_n])

            return mean_squared_error(list_val_std[out_n], y_pred)

        n_trials= 1 + is_optuna_deeplearning*33

        study = optuna.create_study()
        study.optimize(objective_dl, n_trials=n_trials)

        for layers_depth, units_size, keep_prob, patience in \
                                [(
                                study.best_params['layers_depth'],
                                study.best_params['units_size'],
                                study.best_params['keep_prob'],
                                study.best_params['patience']
                                )]:

            batch_size  = 30
            epochs   = 1000
            print('optimized, best deeplearning start')

            dl_params = {'layers_depth': layers_depth,
                        'units_size': units_size,
                        'keep_prob': keep_prob,
                        'patience': patience}
            print(*dl_params)
            model_raw = get_model(**dl_params)
            model_std = get_model(**dl_params)

            model_name =   'dl_best'
            model_name +=  '_depth-'        + str(layers_depth)
            model_name +=  '_unit-'         + str(units_size)
            model_name +=  '_drop-'         + str(keep_prob)
            model_name +=  '_patience-'     + str(patience)

            print('fit the raw data')

            model_raw.fit(list_train_raw[in_n], list_train_raw[out_n],
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1)

            print('fit the std data - please wait')

            model_std.fit(list_train_std[in_n], list_train_std[out_n],
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1)

            print('finish')


            save_regression(model_raw, model_std, model_name)
            model_name = 'DL'
            save_summary(model_raw, model_std, model_name)



        allmodel_results_raw_df.to_csv(os.path.join(parent_path, 'results', theme_name, 'comparison of methods_raw.csv'), index=False)
        allmodel_results_std_df.to_csv(os.path.join(parent_path, 'results', theme_name, 'comparison of methods_std.csv'), index=False)


# settting
# fix the np.random.seed, it can get the same results every time to run this program
np.random.seed(1)
random.seed(1)


def choose_csv():
    #tk_c = tkinter.Tk()
    csv_file_path = tkinter.filedialog.askopenfilename(initialdir = data_processed_path,
                        title = 'choose the csv', filetypes = [('csv file', '*.csv')])

    t_csv_filename.set(str(Path(csv_file_path).name))
    t_csv_filepath.set(csv_file_path)

    t_theme_name.set(Path(csv_file_path).parent.name)

#########   regression by the scikitlearn model ###############
columns_results = ['model_name',
                   'train_model_mse',
                   'train_model_rmse',
                   'test_model_mse',
                   'test_model_rmse',
                   'train_model_score',
                   'test_model_score']
allmodel_results_df = pd.DataFrame(columns = columns_results)
allmodel_bayesian_opt_df = pd.DataFrame()

# tkinter
root = tkinter.Tk()

font1 = font.Font(family='游ゴシック', size=10, weight='bold')
root.option_add("*Font", font1)
style1  = ttk.Style()
style1.configure('my.TButton', font = ('游ゴシック',10)  )
root.title('table-data machine learning')

frame1  = tkinter.ttk.Frame(root, height = 500, width = 500)
frame1.grid(row=0,column=0,sticky=(N,E,S,W))

label_csv  = tkinter.ttk.Label(frame1, text = 'CSVパス:', anchor="w")
t_csv_filename = tkinter.StringVar()
t_csv_filepath = tkinter.StringVar()

entry_csv_filename  = ttk.Entry(frame1, textvariable = t_csv_filename, width = 20)

button_choose_csv    = ttk.Button(frame1, text='CSV選択',
                                 command = choose_csv, style = 'my.TButton')

label_theme_name  = tkinter.ttk.Label(frame1, text = 'テーマ名を入力:')
t_theme_name = tkinter.StringVar()
entry_theme_name  = ttk.Entry(frame1, textvariable = t_theme_name, width = 20)

label_id_clm_num   = tkinter.ttk.Label(frame1, text = '非データの列数を入力:')
label_input_clm_num   = tkinter.ttk.Label(frame1, text = '入力変数の列数を入力:')
label_output_clm_num   = tkinter.ttk.Label(frame1, text = '出力変数の列数を入力:')

t_info_clm_num = tkinter.StringVar()
t_input_clm_num = tkinter.StringVar()
t_output_clm_num = tkinter.StringVar()

entry_id_clm_num  = ttk.Entry(frame1, textvariable = t_info_clm_num, width = 20)
entry_input_clm_num  = ttk.Entry(frame1, textvariable = t_input_clm_num, width = 20)
entry_output_clm_num  = ttk.Entry(frame1, textvariable = t_output_clm_num, width = 20)

save_folder_name = os.path.dirname(t_csv_filename.get()) + 'result'

Booleanvar_sklearn = tkinter.BooleanVar()
Booleanvar_optuna_sklearn = tkinter.BooleanVar()
Booleanvar_deeplearning = tkinter.BooleanVar()
Booleanvar_optuna_deeplearning = tkinter.BooleanVar()

Booleanvar_gridsearch = tkinter.BooleanVar()
Booleanvar_bayesian_opt = tkinter.BooleanVar()

var_bayesian = tkinter.IntVar()
var_bayesian.set(0)

Booleanvar_sklearn.set(True)
Booleanvar_optuna_sklearn.set(False)
Booleanvar_deeplearning.set(False)

Checkbutton_sklearn = tkinter.Checkbutton(frame1, text = '機械学習', variable = Booleanvar_sklearn)
Checkbutton_optuna_sklearn = tkinter.Checkbutton(frame1, text = '高精度', variable = Booleanvar_optuna_sklearn)

Checkbutton_deeplearning = tkinter.Checkbutton(frame1, text = 'ディープラーニング', variable = Booleanvar_deeplearning)
Checkbutton_optuna_deeplearning = tkinter.Checkbutton(frame1, text = '高精度', variable = Booleanvar_optuna_deeplearning)

Checkbutton_gridsearch = tkinter.Checkbutton(frame1, text = '全探索', variable = Booleanvar_gridsearch)
Checkbutton_bayesian_opt = tkinter.Checkbutton(frame1, text = 'ベイズ最適化', variable = Booleanvar_bayesian_opt)

Radiobutton_bayesian_max = tkinter.Radiobutton(frame1, value = 0, text = '最大化', variable = var_bayesian)
Radiobutton_bayesian_min = tkinter.Radiobutton(frame1, value = 1, text = '最小化', variable = var_bayesian)
Radiobutton_bayesian_val = tkinter.Radiobutton(frame1, value = 2, text = '目的値', variable = var_bayesian)

t_bayesian_val = tkinter.StringVar()
entry_bayesian_val = ttk.Entry(frame1, textvariable = t_bayesian_val, width = 5)

button_learning     = ttk.Button(frame1, text='訓練開始',
                                 command = learning, style = 'my.TButton')
# set canvas information

frame2 = tkinter.Toplevel()
frame2.title('graph')
frame2.geometry('1300x900')
frame2.grid()
photo_size = 400

canvas_predicted_values = tkinter.Canvas(frame2, width = photo_size, height = photo_size)
try:
    image_tmp_open = Image.open('logo\logo1.png')
except FileNotFoundError:
    print('logo1.png was not found')
else:
    global image_predicted_values
    image_predicted_values = ImageTk.PhotoImage(image_tmp_open, master=frame2)
    #values is center position
    canvas_predicted_values.create_image(int(photo_size/2), int(photo_size/2), image=image_predicted_values)
    canvas_predicted_values.grid(row=1, column = 1, sticky= W)

canvas_important_variable = tkinter.Canvas(frame2, width = photo_size, height = photo_size)
try:
    image_tmp_open = Image.open('logo\logo2.png')
except FileNotFoundError:
    print('logo2.png was not found')
else:
    global image_important_variable
    image_important_variable = ImageTk.PhotoImage(image_tmp_open, master=frame2)
    #values is center position
    canvas_important_variable.create_image(int(photo_size/2), int(photo_size/2), image=image_important_variable)
    canvas_important_variable.grid(row=1, column = 2, sticky= W)


canvas_correlation_coefficient = tkinter.Canvas(frame2, width = photo_size, height = photo_size)
try:
    image_tmp_open = Image.open('logo\logo3.png')
except FileNotFoundError:
    print('logo3.png was not found')
else:
    global image_correlation_coefficient
    image_correlation_coefficient = ImageTk.PhotoImage(image_tmp_open, master=frame2)
    #values is center position
    canvas_correlation_coefficient.create_image(int(photo_size/2),int(photo_size/2), image=image_correlation_coefficient)
    canvas_correlation_coefficient.grid(row=2, column = 1, sticky= W)

canvas_pairplot = tkinter.Canvas(frame2, width = photo_size, height = photo_size)
try:
    image_tmp_open = Image.open('logo\logo4.png')
except FileNotFoundError:
    print('logo4.png was not found')
else:
    global image_pairplot
    image_pairplot = ImageTk.PhotoImage(image_tmp_open, master=frame2)
    #values is center position
    canvas_pairplot.create_image(int(photo_size/2),int(photo_size/2), image=image_pairplot)
    canvas_pairplot.grid(row=2, column = 2, sticky= W)


canvas_score = tkinter.Canvas(frame2, width = photo_size, height = photo_size)
try:
    image_tmp_open = Image.open('logo\logo5.png')
except FileNotFoundError:
    print('logo6.png was not found')
else:
    global image_score
    image_score = ImageTk.PhotoImage(image_tmp_open, master=frame2)
    #values is center position
    canvas_score.create_image(int(photo_size/2),int(photo_size/2), image=image_score)
    canvas_score.grid(row=1, column = 3, sticky= W)


canvas_contour = tkinter.Canvas(frame2, width = photo_size, height = photo_size)
try:
    image_tmp_open = Image.open('logo\logo6.png')
except FileNotFoundError:
    print('logo6.png was not found')
else:
    global image_contour
    image_contour = ImageTk.PhotoImage(image_tmp_open, master=frame2)
    #values is center position
    canvas_contour.create_image(int(photo_size/2),int(photo_size/2), image=image_contour)
    canvas_contour.grid(row=2, column = 3, sticky= W)





label_csv.grid(row=2,column=1,sticky=E)
entry_csv_filename.grid(row=2,column=2,sticky=W)
button_choose_csv.grid(row=1,column=2,sticky=W)

label_theme_name.grid(row=4,column=1,sticky=E)
entry_theme_name.grid(row=4,column=2,sticky=W)

label_id_clm_num.grid(row=6,column=1,sticky=E)
label_input_clm_num.grid(row=7,column=1,sticky=E)
label_output_clm_num.grid(row=8,column=1,sticky=E)

entry_id_clm_num.grid(row=6,column=2,sticky=W)
entry_input_clm_num.grid(row=7,column=2,sticky=W)
entry_output_clm_num.grid(row=8,column=2,sticky=W)

Checkbutton_sklearn.grid(row = 9, column =2, sticky = W)
Checkbutton_optuna_sklearn.grid(row = 9, column =3, sticky = W)
Checkbutton_deeplearning.grid(row = 10, column = 2, stick = W)
Checkbutton_optuna_deeplearning.grid(row = 10, column = 3, stick = W)
Checkbutton_gridsearch.grid(row = 11, column = 2, stick = W)
Checkbutton_bayesian_opt.grid(row = 12, column = 2, stick = W)

Radiobutton_bayesian_max.grid(row = 12, column = 3)
Radiobutton_bayesian_min.grid(row = 12, column = 4)
Radiobutton_bayesian_val.grid(row = 12, column = 5)
entry_bayesian_val.grid(row = 13, column = 5)

button_learning.grid(row = 14 , column = 2, sticky = W)


for child in frame1.winfo_children():
    child.grid_configure(padx=5, pady=5)
for child in frame2.winfo_children():
    child.grid_configure(padx=5, pady=5)
root.mainloop()
