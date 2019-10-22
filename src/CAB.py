

# programmed by YUKI Horie, Glass Research Center 'yuki.horie@'
# do not use JAPANESE!\
# (c) 2019 Horie Yuki Central Glass

from __future__ import print_function

import sklearn
import sklearn.ensemble

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import _tree
from sklearn.externals.six import StringIO
from sklearn import linear_model
from sklearn import svm
from sklearn.kernel_ridge import KernelRidge
#ver0.20
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split


import numpy as np
import numpy

import pandas as pd
import math
import time
import random
import os
import sys
import itertools
import copy

import matplotlib.pyplot as plt
import seaborn as sns

import GPy
import GPyOpt

try:
    import dtreeviz.trees
    import dtreeviz.shadow
    import dtreeviz
except:
    print('dtreeviz was not found')
    pass


import pydotplus

import xgboost as xgb
#from xgboost import plot_tree # sklearn has also plot_tree, so do not import plot_Tree

import tkinter
import tkinter.filedialog
from tkinter import ttk
from tkinter import N,E,S,W
from tkinter import font

from PIL import ImageTk, Image


# #chkprint
# refer from https://qiita.com/AnchorBlues/items/f7725ba87ce349cb0382
from inspect import currentframe
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


# Graphviz path
#http://spacegomi.hatenablog.com/entry/2018/01/26/170721

desktop_path = os.getenv('HOMEDRIVE') + os.getenv('HOMEPATH') + '\\Desktop'
graphviz_path = desktop_path + '\\graphviz\\release\\bin\\dot.exe'
graphviz_path = os.path.join(desktop_path, 'graphviz', 'release', 'bin', 'dot.exe')

columns_results = ['model_name', 'train_model_mse', 'train_model_rmse', 'test_model_mse', 'test_model_rmse', 'train_model_score', 'test_model_score']
allmodel_results_raw_df = pd.DataFrame(columns = columns_results)
allmodel_results_std_df = pd.DataFrame(columns = columns_results)


# to make the folder of Machine Learning
#if os.path.exists(address_) == False:
#    print('no exist ', address_)
#    pass


# make the folder for saving the results
def chk_mkdir(paths):
    for path_name in paths:
        if os.path.exists(path_name) == False:
            os.mkdir(path_name)
    return

def choose_csv():
    tk_c = tkinter.Tk()
    current_dir = os.getcwd()

    csv_file_path = tkinter.filedialog.askopenfilename(initialdir = current_dir,
    title = 'choose the csv', filetypes = [('csv file', '*.csv')])

    t_csv.set(csv_file_path)


def learning():
    # theme_name used as the result folder name
    theme_name = 'solubility'
    theme_name = t_theme_name.get()

    # cav & theme
    csv_path = t_csv.get()

    # evaluate of all candidate or not
    is_gridsearch = Booleanvar_gridsearch.get()

    # perform deeplearning or not
    is_dl = Booleanvar_deeplearning.get()

    is_bayesian_opt = Booleanvar_bayesian_opt.get()


    # make the save folder
    print('chdir', os.path.dirname(csv_path))
    os.chdir(os.path.dirname(csv_path))


    # make the folder for saving the results
    def chk_mkdir(paths):
        for path_name in paths:
            if os.path.exists(path_name) == False:
                os.mkdir(path_name)
        return


    paths = [ 'results',
            'results' + os.sep + theme_name,
            'results' + os.sep + theme_name +  os.sep + 'sklearn',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'tree',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'importance',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'parameter_raw',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'parameter_std',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'predict_raw',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'predict_stdtoraw',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'traintest_raw',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'traintest_std',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'traintest_stdtoraw',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'scatter_diagram',
            'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'bayesian_opt',
            'results' + os.sep + theme_name +  os.sep + 'deeplearning',
            'results' + os.sep + theme_name +  os.sep + 'deeplearning'+ os.sep + 'h5',
            'results' + os.sep + theme_name +  os.sep + 'deeplearning'+ os.sep + 'traintest',
            'results' + os.sep + theme_name +  os.sep + 'deeplearning'+ os.sep + 'predict']

    paths = [ 'results',
              os.path.join('results' , theme_name),
              os.path.join('results' , theme_name , 'sklearn'),
              os.path.join('results' , theme_name , 'sklearn', 'tree'),
              os.path.join('results' , theme_name , 'sklearn', 'importance'),
              os.path.join('results' , theme_name , 'sklearn', 'parameter_raw'),
              os.path.join('results' , theme_name , 'sklearn', 'parameter_std'),
              os.path.join('results' , theme_name , 'sklearn', 'predict_raw'),
              os.path.join('results' , theme_name , 'sklearn', 'predict_stdtoraw'),
              os.path.join('results' , theme_name , 'sklearn', 'traintest_raw'),
              os.path.join('results' , theme_name , 'sklearn', 'traintest_std'),
              os.path.join('results' , theme_name , 'sklearn', 'traintest_stdtoraw'),
              os.path.join('results' , theme_name , 'sklearn', 'scatter_diagram'),
              os.path.join('results' , theme_name , 'sklearn', 'bayesian_opt'),
              os.path.join('results' , theme_name , 'deeplearning'),
              os.path.join('results' , theme_name , 'deeplearning', 'h5'),
              os.path.join('results' , theme_name , 'deeplearning', 'traintest'),
              os.path.join('results' , theme_name , 'deeplearning', 'predict')]





    chk_mkdir(paths)

    # csv information
    info_num    = int(t_info_clm_num.get())     # information columns in csv file
    input_num   = int(t_input_clm_num.get())    # input data  columns in csv file
    output_num  = int(t_output_clm_num.get())   # output data columns in csv file

    # csv category information
    category_list = []

    input_num_plus = 0
    output_num_plus = 0

    for column_name in category_list:
        column_nth = raw_data_df.columns.get_loc(column_name)
        num_plus = raw_data_df[column_name].nunique()
        #print(num_plus)

        if column_nth <= info_num + input_num:
            input_num_plus += num_plus-1
        elif column_nth < info_num + input_num:
            output_num_plus += num_plus-1

        pass

    input_num   += input_num_plus
    output_num  += output_num_plus
    list_num    = [input_num, output_num]

    def get_ordinal_mapping(obj):

        listx = list()
        for x in obj.category_mapping:
            listx.extend([tuple([x['col']])+ i for i in x['mapping']])
        df_ord_map = pd.DataFrame(listx)
        return df_ord_map

    try:
        import category_encoders

        ce_onehot   = category_encoders.OneHotEncoder(cols = category_list, handle_unknown = 'impute')
        ce_binary   = category_encoders.BinaryEncoder(cols = category_list, handle_unknown = 'impute')

        ce_onehot.fit_transform(raw_data_df)
        raw_data_df = ce_binary.fit_transform(raw_data_df)

        #get_ordinal_mapping(ce_onehot)
        #print(get_ordinal_mapping(ce_onehot))

    except:
        print('Error - no import category_encoders')
        pass

    #raw_data_df = pd.read_csv(open(str(address_) + str(CSV_NAME) ,encoding="utf-8_sig"))
    try:
        raw_data_df = pd.read_csv(open(csv_path ,encoding="utf-8_sig"))
        print('utf-8で読み込みました')
    except:
        raw_data_df = pd.read_csv(open(csv_path ,encoding="shift-jis"))
        print('shift-jisで読み込みました')

    # csv information data columns
    info_col    = info_num
    input_col   = info_num + input_num
    output_col  = info_num + input_num + output_num


    info_raw_df         = raw_data_df.iloc[:, 0         : info_col]
    input_raw_df        = raw_data_df.iloc[:, info_col  : input_col]
    output_raw_df       = raw_data_df.iloc[:, input_col : output_col]
    in_output_raw_df    = raw_data_df.iloc[:, info_col  : output_col]
    list_df             = [input_raw_df, output_raw_df]

    info_feature_names              = info_raw_df.columns
    input_feature_names             = input_raw_df.columns
    output_feature_names            = output_raw_df.columns
    list_feature_names              = [input_feature_names, output_feature_names]

    predict_input_feature_names     = list(map(lambda x:x + '-predict' , input_feature_names))
    predict_output_feature_names    = list(map(lambda x:x + '-predict' , output_feature_names))
    list_predict_feature_names      = [predict_input_feature_names, predict_output_feature_names]

    from sklearn.preprocessing import StandardScaler
    in_output_sc_model  = StandardScaler()
    input_sc_model      = StandardScaler()
    output_sc_model     = StandardScaler()
    list_sc_model       = [input_sc_model, output_sc_model]

    in_output_std_df    = pd.DataFrame(in_output_sc_model.fit_transform(in_output_raw_df))
    input_std_df        = pd.DataFrame(input_sc_model.fit_transform(input_raw_df))
    output_std_df       = pd.DataFrame(output_sc_model.fit_transform(output_raw_df))

    input_raw_des   = input_raw_df.describe()
    output_raw_des  = output_raw_df.describe()

    input_raw_max   = input_raw_des.loc['max']
    input_raw_min   = input_raw_des.loc['min']
    output_raw_max  = output_raw_des.loc['max']
    output_raw_min  = output_raw_des.loc['min']

    list_raw_max    = [input_raw_max, output_raw_max]
    list_raw_min    = [input_raw_min, output_raw_min]

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
    train_std_df, test_std_df   = train_test_split(in_output_std_df, test_size=0.2)
    np.random.seed(10)
    random.seed(10)
    train_raw_df, test_raw_df   = train_test_split(in_output_raw_df, test_size=0.2)

    # transform from pandas dataframe to numpy array
    train_raw_np = np.array(train_raw_df)
    test_raw_np  = np.array(test_raw_df)
    train_std_np = np.array(train_std_df)
    test_std_np  = np.array(test_std_df)

    # split columns to info, input, output
    [train_input_raw, train_output_raw] = np.hsplit(train_raw_np, [input_num])
    list_train_raw                      = [train_input_raw, train_output_raw]
    [test_input_raw,  test_output_raw]  = np.hsplit(test_raw_np,  [input_num])
    list_test_raw                       = [test_input_raw, test_output_raw]

    [train_input_std, train_output_std] = np.hsplit(train_std_np, [input_num])
    list_train_std                      = [train_input_std  , train_output_std]

    [test_input_std,  test_output_std]  = np.hsplit(test_std_np , [input_num])
    list_test_std                       = [test_input_std   , test_output_std]

    train_input_raw_df                  = pd.DataFrame(train_input_raw  , columns = input_feature_names)
    test_input_raw_df                   = pd.DataFrame(test_input_raw   , columns = input_feature_names)
    train_output_raw_df                 = pd.DataFrame(train_output_raw , columns = output_feature_names)
    test_output_raw_df                  = pd.DataFrame(test_output_raw  , columns = output_feature_names)

    list_train_raw_df                   = [train_input_raw_df, train_output_raw_df]
    list_test_raw_df                    = [test_input_raw_df, test_output_raw_df]

    train_input_std_df                  = pd.DataFrame(train_input_std  , columns = input_feature_names)
    test_input_std_df                   = pd.DataFrame(test_input_std   , columns = input_feature_names)
    train_output_std_df                 = pd.DataFrame(train_output_std , columns = output_feature_names)
    test_output_std_df                  = pd.DataFrame(test_output_std  , columns = output_feature_names)

    list_train_std_df                   = [train_input_std_df , train_output_std_df ]
    list_test_std_df                    = [test_input_std_df  , test_output_std_df ]

    plt.figure(figsize=(5,5))
    sns.heatmap(in_output_raw_df.corr(), cmap = 'Oranges',annot=False, linewidths = .5)
    #plt.savefig('results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep +'correlation_coefficient.png', dpi = 240)
    plt.savefig(os.path.join('results' , theme_name , 'sklearn','correlation_coefficient.png'), dpi = 240)

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
                    #print("{}if {} <= {}:".format(indent, name, threshold))
                    #print("{}if {} <= {}:".format(indent, name, threshold), file=f)
                    recurse(tree_.children_left[node], depth + 1)
                    #print("{}else:  # if {} > {}".format(indent, name, threshold), file=f)
                    #print("{}else:  # if {} > {}".format(indent, name, threshold))
                    recurse(tree_.children_right[node], depth + 1)
                else:
                    #print("{}return {}".format(indent, tree_.value[node]))
                    #print("{}return {}".format(indent, tree_.value[node]), file=f)
                    pass

        recurse(0, 1)

    def gridsearch_predict(model_raw,model_std, model_name):
        start_time = time.time()


        gridsearch_predict_raw_df   = pd.DataFrame(model_raw.predict(gridsearch_input_raw), columns = list_predict_feature_names[out_n])

        gridsearch_predict_std_df   = pd.DataFrame(model_std.predict(gridsearch_input_std), columns = list_predict_feature_names[out_n])
        gridsearch_predict_stdtoraw_df       = pd.DataFrame(list_sc_model[out_n].inverse_transform(gridsearch_predict_std_df), columns = list_predict_feature_names[out_n])

        end_time = time.time()
        total_time = end_time - start_time
        #print(model_name, ' ' , total_time)

        #gridsearch_predict_df['Tmax-']      = gridsearch_predict_df.max(axis = 1)
        #gridsearch_predict_df['Tdelta-']    = gridsearch_predict_df.min(axis = 1)
        #gridsearch_predict_df['Tmin-']      = gridsearch_predict_df['Tmax-'] - gridsearch_predict_df['Tdelta-']


        gridsearch_predict_raw_df       = pd.concat([gridsearch_input_raw_df, gridsearch_predict_raw_df], axis = 1)
        gridsearch_predict_stdtoraw_df  = pd.concat([gridsearch_input_std_df, gridsearch_predict_stdtoraw_df], axis = 1)

        '''
        gridsearch_predict_raw_df.to_csv( 'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'predict_raw'+ os.sep
                                     + str(model_name) + '_predict.csv')
        gridsearch_predict_stdtoraw_df.to_csv( 'results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'predict_stdtoraw'+ os.sep
                                     + str(model_name) + '_predict.csv')
        '''
        gridsearch_predict_raw_df.to_csv(     os.path.join('results', theme_name, 'sklearn', 'predict_raw', str(model_name), '_predict.csv'))
        gridsearch_predict_stdtoraw_df.to_csv(os.path.join('results', theme_name, 'sklearn', 'predict_stdtoraw', str(model_name), '_predict.csv'))


        return

    #########   search by hyperopt ###############
    #import hyperopt

    #########   save the tree to pdf ###############

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

        save_regression(model_raw, model_std, model_name)

        return


    def save_regression(model_raw, model_std, model_name):

        def save_tree_topdf(model, model_name):

            dot_data = StringIO()
            try:
                #sklearn.tree.export_graphviz(model, out_file=dot_data, feature_names = list_feature_names[in_n].replace('/','_'))
                sklearn.tree.export_graphviz(model, out_file=dot_data, feature_names = list_feature_names[in_n])
            except:
                #xgb.to_graphviz(model,  out_file=dot_data, feature_names = list_feature_names[in_n].replace('/','_'))
                xgb.to_graphviz(model,  out_file=dot_data, feature_names = list_feature_names[in_n])
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

            # refer from https://qiita.com/wm5775/items/1062cc1e96726b153e28
            # the Graphviz2.38 dot.exe
            graph.progs = {'dot':graphviz_path}

            #graph.write_pdf('results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'tree'+ os.sep
            #                 + model_name + '.pdf')

            graph.write_pdf(os.path.join('results', theme_name, 'sklearn', 'tree', model_name, '.pdf')

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
            #plt.show()
            #plt.savefig('results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'importance' + os.sep
            #             + str(model_name)   + '.png', dpi = 240)
            plt.savefig(os.path.join('results', theme_name, 'sklearn', 'importance', str(model_name), '.png'), dpi = 240)
            return


        def save_scatter_diagram(model, model_name):

            pred_train = model.predict(list_train_raw[in_n])
            pred_test = model.predict(list_test_raw[in_n])

            from PIL import Image

            plt.figure(figsize=(5,5))
            plt.scatter(list_train_raw[out_n], pred_train, label = 'Train', c = 'blue')
            plt.title("-" + model_name)
            plt.xlabel('Measured value')
            plt.ylabel('Predicted value')
            plt.scatter(list_test_raw[out_n], pred_test, c = 'lightgreen', label = 'Test', alpha = 0.8)
            plt.legend(loc = 4)

            #plt.savefig('results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'scatter_diagram' + os.sep +  str(model_name) + '_scatter.png')
            plt.savefig(os.path.join('results', theme_name, 'sklearn', 'scatter_diagram',  str(model_name), '_scatter.png'))



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

        #chkprint(model_name)


        train_result_raw_df.to_csv(os.path.join('results', theme_name, 'sklearn', 'traintest_raw', str(model_name), '_train_raw.csv'))
        test_result_raw_df.to_csv(os.path.join('results', theme_name, 'sklearn', 'traintest_raw', str(model_name), '_test_raw.csv'))
        train_result_std_df.to_csv(os.path.join('results', theme_name, 'sklearn', 'traintest_std', str(model_name), '_train_std.csv'))
        test_result_std_df.to_csv(os.path.join('results', theme_name, 'sklearn', 'traintest_std', str(model_name), '_test_std.csv'))
        train_result_stdtoraw_df.to_csv(os.path.join('results', theme_name, 'sklearn', 'traintest_stdtoraw', str(model_name),  '_train_stdtoraw.csv'))
        test_result_stdtoraw_df.to_csv(os.path.join('results', theme_name, 'sklearn', 'traintest_stdtoraw', str(model_name),  '_test_stdtoraw.csv'))

        save_scatter_diagram(model_raw, model_name + '_raw')
        save_scatter_diagram(model_std, model_name + '_std')


        if hasattr(model_std, 'get_params') == True:
            model_params        = model_std.get_params()
            params_df           = pd.DataFrame([model_std.get_params])

        if hasattr(model_raw, 'intercept_') == True &  hasattr(model_std, 'coef_') == True:
            model_intercept_raw_df  = pd.DataFrame(model_raw.intercept_)
            model_coef_raw_df       = pd.DataFrame(model_raw.coef_)
            model_parameter_raw_df  = pd.concat([model_intercept_raw_df, model_coef_raw_df])
            model_parameter_raw_df.to_csv(os.path.join()'results', theme_name, 'sklearn', 'parameter_raw', str(model_name), '_parameter.csv')

        if hasattr(model_std, 'intercept_') == True &  hasattr(model_std, 'coef_') == True:
            model_intercept_std_df  = pd.DataFrame(model_std.intercept_)
            model_coef_std_df       = pd.DataFrame(model_std.coef_)
            model_parameter_std_df  = pd.concat([model_intercept_std_df, model_coef_std_df])
            model_parameter_std_df.to_csv(os.path.join()'results', theme_name, 'sklearn', 'parameter_std', str(model_name), '_parameter.csv')

        if hasattr(model_raw, 'tree_') == True:
            save_tree_topdf(model_raw, model_name)
            pass


        if hasattr(model_raw, 'feature_importances_') == True:

            importances = pd.Series(model_raw.feature_importances_)
            importances = np.array(importances)
            #print(importances)
            #importances = importances.sort_values()

            label       = list_feature_names[in_n]

            # initialization of plt
            plt.clf()

            plt.bar(label, importances)

            plt.xticks(rotation=90)
            plt.xticks(fontsize=8)
            plt.rcParams["font.size"] = 12

            plt.title("importance in the tree " + str(theme_name))
            #plt.show()
            #plt.savefig('results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'importance'+ os.sep
            #             + str(model_name)  + '.png', dpi = 240)
            plt.savefig(os.path.join('results', theme_name, 'sklearn', 'importance', str(model_name), '.png'), dpi = 240)


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


        # bayesian Optimization
        # refer https://qiita.com/shinmura0/items/2b54ab0117727ce007fd
        # refer https://qiita.com/marshi/items/51b82a7b990d51bd98cd

        if is_bayesian_opt == True:
            print('start the bayesian optimaization')
            def function_for_bayesian(x):

                optimize_type = var_bayesian.get()
                if optimize_type == 0:
                    #max
                    return model_std.predict(x) * -1
                elif optimize_type == 1:
                    #max
                    return model_std.predict(x)
                elif optimize_type ==2 and t_bayesian_val.get != '':
                    target_std_value = list_sc_model[out_n].transform(t_bayesian_val.get())

                    return (model_std.predict(x) - target_std_value)**2
                else:
                    return model_std.predict(x)



            if inv_ == 0 and list_num[out_n] ==1 :
                bounds = []
                print(list_num[in_n])
                for i in range(list_num[in_n]):
                    bounds.append({'name': list_feature_names[in_n][i] , 'type': 'continuous', 'domain': (list_std_min[in_n][i],list_std_max[in_n][i])})

                #chkprint(bounds)
                myBopt = GPyOpt.methods.BayesianOptimization(f=function_for_bayesian, domain=bounds)

                myBopt.run_optimization(max_iter=10)

                print('result of bayesian optimization')
                print(list_sc_model[in_n].inverse_transform(np.array([myBopt.x_opt])))
                print(list_sc_model[out_n].inverse_transform(np.array([myBopt.fx_opt])))
                optimized_input_df = pd.DataFrame(list_sc_model[in_n].inverse_transform(np.array([myBopt.x_opt])), columns = list_feature_names[in_n])
                optimized_output_df = pd.DataFrame(list_sc_model[out_n].inverse_transform(np.array([myBopt.fx_opt])),columns = list_feature_names[out_n])
                print(model_name)
                model_name_df = pd.Series(model_name)
                optimized_result_df = pd.concat([model_name_df, optimized_input_df, optimized_output_df], axis =1)
                #optimized_result_df.to_csv('results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep + 'bayesian_opt' + os.sep
                #                     + str(model_name) + '_bayesian_result.csv')
                optimized_result_df.to_csv(os.path.join('results', theme_name, 'sklearn', 'bayesian_opt', str(model_name), '_bayesian_result.csv'))

                allmodel_bayesian_opt_df = pd.concat([allmodel_bayesian_opt_df, optimized_result_df])

                #sys.exit()

            return


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

        #print(in_n)
        #print(out_n)

        direction_name_list = ['normal', 'inverse']
        direction_name      = direction_name_list[inv_]

        columns_results = ['model_name', 'train_model_mse', 'train_model_rmse', 'test_model_mse', 'test_model_rmse', 'train_model_score', 'test_model_score']
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

        print('list_train_raw[in_n]')

        print(list_train_raw[in_n])
        model.fit(list_train_raw[in_n], list_train_raw[out_n])

        importances         = np.array(model.feature_importances_)
        label       = list_feature_names[in_n]

        pred_train = model.predict(list_train_raw[in_n])
        pred_test = model.predict(list_test_raw[in_n])

        from PIL import Image

        plt.figure(figsize=(5,5))
        plt.scatter(list_train_raw[out_n], pred_train, label = 'Train', c = 'blue')
        plt.title("-" + model_name)
        plt.title('Mordred predict')
        plt.xlabel('Measured value')
        plt.ylabel('Predicted value')
        plt.scatter(list_test_raw[out_n], pred_test, c = 'lightgreen', label = 'Test', alpha = 0.8)
        plt.legend(loc = 4)

        #plt.savefig('results' + os.sep + theme_name +  os.sep + 'sklearn'+ os.sep +'meas_pred.png')
        plt.savefig(os.path.join('results', theme_name, 'sklearn', 'meas_pred.png'))

        img1 = Image.open(os.path.join('results', theme_name, 'sklearn', 'meas_pred.png'))

        img1_resize = img1.resize((photo_size, photo_size), Image.LANCZOS)
        img1_resize.save(os.path.join('results', theme_name, 'sklearn', 'meas_pred.png'))

        global image_predicted_values
        image_open = Image.open(os.path.join('results', theme_name, 'sklearn', 'meas_pred.png'))
        image_predicted_values = ImageTk.PhotoImage(image_open, master=frame2)

        canvas_predicted_values.create_image(int(photo_size/2),int(photo_size/2), image=image_predicted_values)


        ########################
        plt.figure(figsize =(5,5))
        plt.bar(label, importances)

        plt.xticks(rotation=90)
        plt.xticks(fontsize=8)
        plt.rcParams["font.size"] = 12

        plt.title("-" + model_name)
        #plt.show()
        plt.savefig(os.path.join('results', theme_name, 'sklearn', 'tmp_importances.png'), dpi = 240)

        img2 = Image.open(os.path.join('results', theme_name, 'sklearn', 'tmp_importances.png'))

        img2_resize = img2.resize((photo_size, photo_size), Image.LANCZOS)
        img2_resize.save(os.path.join('results', theme_name, 'sklearn', 'tmp_importances.png'))

        global image_important_variable
        image_open = Image.open(os.path.join('results', theme_name, 'sklearn', 'tmp_importances.png'))
        image_important_variable = ImageTk.PhotoImage(image_open, master=frame2)

        canvas_important_variable.create_image(int(photo_size/2),int(photo_size/2), image=image_important_variable)



        global image_correlation_coefficient

        img3 = Image.open(os.path.join('results', theme_name, 'sklearn', 'correlation_coefficient.png'))
        img3_resize = img3.resize((photo_size, photo_size), Image.LANCZOS)
        img3_resize.save(os.path.join('results', theme_name, 'sklearn', 'correlation_coefficient.png'))
        image_open = Image.open(os.path.join('results', theme_name, 'sklearn', 'correlation_coefficient.png'))

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



        ##################### Linear Regression #####################

        model = linear_model.LinearRegression()
        model_name = 'Linear_Regression_'

        fit_model_std_raw(model, model_name)
        LinearRegression_model = model
        LinearRegression_model_name = model_name
        #LinearRegression_test_r2_score = r2_score(model_std.predict(test_input_std), test_output_std)

        ##################### Regression of Stochastic Gradient Descent #####################
        max_iter = 1000

        model = MultiOutputRegressor(linear_model.SGDRegressor(max_iter = max_iter))
        model_name = 'MultiOutput Stochastic Gradient Descent_'
        model_name += 'max_iter_'+str(max_iter)

        fit_model_std_raw(model, model_name)
        Multi_SGD_model = model
        Multi_SGD_model_name = model_name


        ##################### Regression of SVR #####################
        kernel_ = 'rbf'
        C_= 1

        model = MultiOutputRegressor(svm.SVR(kernel = kernel_, C = C_))
        model_name = 'MultiOutput SupportVectorRegressor_'
        model_name += 'kernel_'+str(kernel_)
        model_name += 'C_'+str(C_)

        fit_model_std_raw(model, model_name)
        Multi_SVR_model = model
        Multi_SVR_model_name = model_name

        # refer https://www.slideshare.net/ShinyaShimizu/ss-11623505


        ##################### Regression of Ridge #####################
        tmp_r2_score = 0
        for alpha_ in [0.01, 0.1, 1.0] :
            model = linear_model.Ridge(alpha = alpha_)
            model_name = 'Ridge_'
            model_name += 'alpha_'+str(alpha_)

            fit_model_std_raw(model, model_name)




        ##################### Regression of KernelRidge #####################
        alpha_ = 1.0
        model = KernelRidge(alpha=alpha_, kernel='rbf')
        model_name = 'KernelRidge_'
        model_name += 'alpha_'+str(alpha_)

        fit_model_std_raw(model, model_name)

        KernelRidge_model = model
        KernelRidge_model_name = model_name

        ##################### Regression of Lasso #####################
        tmp_r2_score = 0
        for alpha_ in [0.01, 0.1, 1.0] :
            model = linear_model.Lasso(alpha = alpha_)
            model_name = 'Lasso_'
            model_name += 'alpha_'+str(alpha_)

            fit_model_std_raw(model, model_name)

        ##################### Regression of Elastic Net #####################

        alpha_      = [0.01, 0.1]
        l1_ratio_   = [0.25, 0.75]

        tmp_r2_score = 0
        for alpha_, l1_ratio_ in itertools.product(alpha_, l1_ratio_):

            model = linear_model.ElasticNet(alpha=alpha_, l1_ratio = l1_ratio_)
            model_name = 'ElasticNet_'
            model_name += 'alpha_'+str(alpha_)
            model_name += 'l1_ratio_'+str(l1_ratio_)

            fit_model_std_raw(model, model_name)



        ##################### Regression of MultiTaskLassoCV #####################
        max_iter_ = 1000

        model = linear_model.MultiTaskLassoCV()
        model_name = 'MultiTaskLasso_'
        model_name += 'max_iter_'+str(max_iter)

        fit_model_std_raw(model, model_name)

        ##################### Regression of Multi Task Elastic Net CV #####################
        model = linear_model.MultiTaskElasticNetCV()

        model_name = 'MTElasticNet_'
        fit_model_std_raw(model, model_name)


        ##################### Regression of OrthogonalMatchingPursuit #####################
        #model = linear_model.OrthogonalMatchingPursuit()
        #model_name = 'OrthogonalMatchingPursuit_'

        #fit_model_std_raw(model, model_name)

        ##################### Regression of BayesianRidge #####################
        model = MultiOutputRegressor(linear_model.BayesianRidge())
        model_name = 'MultiOutput BayesianRidge_'

        fit_model_std_raw(model, model_name)

        ##################### Regression of PassiveAggressiveRegressor #####################
        #model = MultiOutputRegressor(linear_model.PassiveAggressiveRegressor())
        #model_name = 'MultiOutput PassiveAggressiveRegressor_'

        #fit_model_std_raw(model, model_name)

        ##################### Regression of PolynomialFeatures #####################
        '''
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import Pipeline
        # http://techtipshoge.blogspot.com/2015/06/scikit-learn.html
        # http://enakai00.hatenablog.com/entry/2017/10/13/145337

        for degree in [2]:
            model = Pipeline([
                ('poly', PolynomialFeatures(degree = 2),
                'linear', MultiOutputRegressor(linear_model.LinearRegression()))
                ])

            model_name = 'PolynomialFeatures_'
            model_name += 'degree_' + str(degree)

            fit_model_std_raw(model, model_name)
        '''

        ##################### Regression of GaussianProcessRegressor #####################
        '''
        from sklearn.gaussian_process import GaussianProcessRegressor

        model = MultiOutputRegressor(GaussianProcessRegressor())
        model_name = 'MultiOutput GaussianProcessRegressor_'

        fit_model_std_raw(model, model_name)
        '''

        ##################### Regression of GaussianNB #####################

        '''
        from sklearn.naive_bayes import GaussianNB

        model = MultiOutputRegressor(GaussianNB())
        model_name = 'MultiOutput GaussianNB_'

        fit_model_std_raw(model, model_name)
        '''

        ##################### Regression of GaussianNB #####################

        '''
        from sklearn.naive_bayes import  ComplementNB

        model = ComplementNB()
        model_name = 'ComplementNB_'

        fit_model_std_raw(model, model_name)
        '''

        ##################### Regression of MultinomialNB #####################

        '''
        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()
        model_name = 'MultinomialNB_'

        fit_model_std_raw(model, model_name)
        '''

        ##################### Regression of DecisionTreeRegressor #####################
        tmp_r2_score = 0
        for max_depth in [7,10]:
            model = sklearn.tree.DecisionTreeRegressor(max_depth = max_depth)
            model_name = 'DecisionTreeRegressor_'
            model_name += 'max_depth_'+str(max_depth)

            fit_model_std_raw(model, model_name)


        ##################### Regression of Multioutput DecisionTreeRegressor #####################
        tmp_r2_score = 0
        for max_depth in [3,5,9]:

            model = MultiOutputRegressor(sklearn.tree.DecisionTreeRegressor(max_depth = max_depth))
            model_name = 'MultiOutput DecisionTreeRegressor_'
            model_name += 'max_depth_' + str(max_depth)
            fit_model_std_raw(model, model_name)
            '''
            if r2_score(model_std.predict(test_input_std), test_output_std) > tmp_r2_score:
                tmp_r2_score =  r2_score(model_std.predict(test_input_std), test_output_std)
                Multi_DecisionTreeRegressor_model  = model
                Multi_DecisionTreeRegressor_model_name = model_name
            '''
        #################### Regression of RandomForestRegressor #####################
        tmp_r2_score = 0
        for max_depth in [3,5,7,9,11]:
            model = sklearn.ensemble.RandomForestRegressor(max_depth = max_depth)
            model_name = ''
            model_name += 'RandomForestRegressor_'
            #model_name += get_variablename(max_depth)
            model_name += 'max_depth_'+str(max_depth)

            fit_model_std_raw(model, model_name)
            '''
            if r2_score(model_std.predict(test_input_std), test_output_std) > tmp_r2_score:
                tmp_r2_score =  r2_score(model_std.predict(test_input_std), test_output_std)
                RandomForestRegressor_model  = model
                RandomForestRegressor_model_name = model_name
            '''
        ##################### Regression of XGBoost #####################
        # refer from https://github.com/FelixNeutatz/ED2/blob/23170b05c7c800e2d2e2cf80d62703ee540d2bcb/src/model/ml/CellPredict.py

        estimator__min_child_weight_ = [5] #1,3
        estimator__subsample_        = [0.9] #0.7, 0.8,
        estimator__learning_rate_    = [0.1,0.01] #0.1
        estimator__max_depth_        = [7]
        estimator__n_estimators_      = [100]

        tmp_r2_score = 0
        for estimator__min_child_weight, estimator__subsample, estimator__learning_rate, estimator__max_depth, estimator__n_estimators \
            in itertools.product(estimator__min_child_weight_, estimator__subsample_, estimator__learning_rate_, estimator__max_depth_,estimator__n_estimators_ ):

            xgb_params = {'estimator__min_child_weight': estimator__min_child_weight,
                        'estimator__subsample': estimator__subsample,
                        'estimator__learning_rate': estimator__learning_rate,
                        'estimator__max_depth': estimator__max_depth,
                        'estimator__n_estimators': estimator__n_estimators,
                        'colsample_bytree': 0.8,
                        'silent': 1,
                        'seed': 0,
                        'objective': 'reg:linear'}

            model = MultiOutputRegressor(xgb.XGBRegressor(**xgb_params))

            model_name = 'MultiOutput-XGBoost'
            model_name += 'min_child_weight_'+str(estimator__min_child_weight)
            model_name += 'subsample_'+str(estimator__subsample)
            model_name += 'learning_rate_'+str(estimator__learning_rate)
            model_name += 'max_depth_'+str(estimator__max_depth)
            model_name += 'n_estimators_'+str(estimator__n_estimators)

            fit_model_std_raw(model, model_name)
            '''
            if r2_score(model_std.predict(test_input_std), test_output_std) > tmp_r2_score:
                tmp_r2_score =  r2_score(model_std.predict(test_input_std), test_output_std)
                Xgboost_model  = model
                Xgboost_model_name = model_name
            '''
        allmodel_r2_score_list = [
            [LinearRegression_model_name, LinearRegression_model_name],
            [Multi_SGD_model_name, Multi_SGD_model]

        ]
        allmodel_r2_score_df = pd.DataFrame()


        ################# to csv ##############################
        allmodel_results_raw_df.to_csv(os.path.join('results', theme_name, 'comparison of methods_raw.csv'))
        allmodel_results_std_df.to_csv(os.path.join('results', theme_name, 'comparison of methods_std.csv'))
        allmodel_bayesian_opt_df.to_csv(os.path.join('results', theme_name, 'bayesian_op.csv'))
        #######################################################


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
        pass
    elif is_dl == True :


        import keras
        from keras.models import Sequential
        from keras.layers import InputLayer
        from keras.layers import Dense, Dropout, Flatten
        from keras.layers import Conv2D, MaxPooling2D
        from keras import backend as K
        from keras.layers.normalization import BatchNormalization
        from keras.layers import Activation
        import keras.models
        from keras.wrappers.scikit_learn import KerasRegressor

        from keras.models import load_model
        from keras.utils import plot_model

        import h5py


        def get_model(layers_depth, units_size, keep_prob, patience):

            model =Sequential()
            #model.add(InputLayer(input_shape=(input_num,)))
            # 1Layer
            model.add(Dense(units_size, input_shape=(input_num,)))
            model.add(Activation('relu'))
            model.add(BatchNormalization(mode=0))

            for i in range(layers_depth-2):
                model.add(Dense(units_size))
                model.add(Activation('relu'))
                model.add(BatchNormalization(mode=0))

            model.add(Dense(output_num))


            model.compile(loss='mse',
                    optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])

            return model

        def get_model2(layers_depth, units_size, keep_prob, patience):
            model = Sequential()
            model.add(Dense(input_num, input_dim = input_num, activation = 'relu'))

            for i in range(layers_depth-2):
                model.add(Dense(units_size, activation = 'relu'))
                model.add(BatchNormalization(mode=0))

            model.add(Dense(output_num))

            model.compile(loss = 'mean_squared_error', optimizer = 'adam')

            return model
        # refer from https://github.com/completelyAbsorbed/ML/blob/0ca17d25bae327fe9be8e3639426dc86f3555a5a/Practice/housing/housing_regression_NN.py


        layers_depth  = [4,5]
        units_size  = [64, 32, 16]
        #bn_where    = [3, 0]
        ac_last     = [0, 1]
        keep_prob   = [0]
        patience    = [3000]


        for dp_params in itertools.product(layers_depth, units_size, keep_prob, patience):
            layers_depth, units_size, keep_prob, patience = dp_params

            batch_size  = 30
            epochs   = 10
            cb = keras.callbacks.EarlyStopping(monitor = 'loss'   , min_delta = 0,
                                        patience = patience, mode = 'auto')

            print('input_num', input_num)
            print('output_num', output_num)

            model = KerasRegressor(build_fn = get_model2(*dp_params), nb_epoch=5000, batch_size=5, verbose=0, callbacks=[cb])
            #model.summary()

            model_name =   'deeplearning'
            model_name +=  '_depth-'        + str(layers_depth)
            model_name +=  '_unit-'         + str(units_size)
            #model_name +=  '_BN- '         + str(bn_where)
            #model_name +=  '_AC-'          + str(ac_last)
            model_name +=  '_drop-'         + str(keep_prob)
            model_name +=  '_patience-'     + str(patience)

            model.fit(list_train_std[in_n], list_train_std[out_n],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(list_test_raw[in_n], list_test_raw[out_n]),
                        callbacks=[cb])

            save_regression(model, model_name, list_train_std[in_n], test_input_std)


            #fit_model_std_raw(model, model_name)

        allmodel_results_df.to_csv('comparison of methods.csv')

        ##epochs = 100000
        #batch_size = 32
        '''
        for patience_ in [100,3000]:

            es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience_, verbose=0, mode='auto')

            for layers_depth in [4,3,2]:
                if layers_depth !=1:
                    for units_size in[1024,512,256,128,64,32,16]:
                        for bn_where in [3,0,1,2]:
                            for ac_last in [0,1]:
                                for keep_prob in [0,0.1,0.2]:

                                    model =get_model(layers_depth,units_size,bn_where,ac_last,keep_prob, patience)
                                    #model = KerasRegressor(build_fn = model, epochs=5000, batch_size=5, verbose=0, callbacks=[es_cb])

                                    if units_size >= 1024:
                                        batch_size = 30
                                    elif layers_depth >= 4:
                                        batch_size = 30
                                    elif bn_where ==3:
                                        batch_size=30

                                    else:
                                        batch_size = 30

                                    model_name = "deeplearning"
                                    model_name +=  '_numlayer-'       + str(layers_depth)
                                    model_name +=  '_layersize-'      + str(units_size)
                                    model_name +=  '_bn- '            + str(bn_where)
                                    model_name +=  '_ac-'             + str(ac_last)
                                    model_name +=  '_k_p-'            + str(keep_prob)
                                    model_name +=  '_pat-'       + str(patience_)

                                    fit_model_std_raw(model, model_name)




                                    model.fit(train_input, train_output,
                                            batch_size=batch_size,
                                            epochs=epochs,
                                            verbose=1,
                                            validation_data=(test_input, test_output),
                                            callbacks=[es_cb])

                                    save_regression(model, model_name, list_train_std[in_n], test_input_std)


                                    score_test = model.evaluate(test_input, test_output, verbose=1)
                                    score_train = model.evaluate(train_input, train_output, verbose=1)
                                    test_predict = model.predict(test_input, batch_size=32, verbose=1)
                                    train_predict = model.predict(train_input, batch_size=32, verbose=1)

                                    df_test_pre = pd.DataFrame(test_predict)
                                    df_train_pre = pd.DataFrame(train_predict)

                                    df_test_param = pd.DataFrame(test_output)
                                    df_train_param = pd.DataFrame(train_output)

                                    df_dens_test = pd.concat([df_test_param, df_test_pre], axis=1)
                                    df_dens_train = pd.concat([df_train_param, df_train_pre], axis=1)

                                    savename = ""
                                    savename +=  '_score_train-'    + str("%.3f" % round(score_train[0],3))
                                    savename +=  '_score_test-'     + str("%.3f" % round((score_test[0]),3))
                                    savename +=  '_numlayer-'       + str(layers_depth)
                                    savename +=  '_layersize-'      + str(units_size)
                                    savename +=  '_bn- '            + str(bn_where)
                                    savename +=  '_ac-'             + str(ac_last)
                                    savename +=  '_k_p-'            + str(keep_prob)
                                    savename +=  '_patience-'       + str(patience_)

                                    df_dens_test.to_csv('deeplearning/traintest/' + savename + '_test.csv')
                                    df_dens_train.to_csv('deeplearning/traintest/' + savename + '_train.csv')


                                    model.save('deeplearning/h5/' + model_name + '.h5')


                                    model.summary()



                                    ### evaluation of deeplearning ###
                                    def eval_bydeeplearning(input):
                                        output_predict = model.predict(input, batch_size = 1, verbose= 1)
                                        output_predict = np.array(output_predict)

                                        return output_predict

                                    if is_gridsearch == True:

                                        gridsearch_output = eval_bydeeplearning(gridsearch_input)


                                        #print('start the evaluation by deeplearning')
                                        #print('candidate is ', candidate_number)

                                        start_time = time.time()

                                        iter_deeplearning_predict_df = pd.DataFrame(gridsearch_output, columns = predict_output_feature_names)
                                        iter_deeplearning_predict_df['Tmax'] = iter_deeplearning_predict_df.max(axis=1)
                                        iter_deeplearning_predict_df['Tmin'] = iter_deeplearning_predict_df.min(axis=1)
                                        iter_deeplearning_predict_df['Tdelta'] = iter_deeplearning_predict_df['Tmax'] -  iter_deeplearning_predict_df['Tmin']

                                        iter_deeplearning_df = pd.concat([gridsearch_input_std_df, iter_deeplearning_predict_df], axis=1)

                                        end_time = time.time()

                                        total_time = end_time - start_time
                                        #print('total_time 1', total_time)

                                        predict_df_s = iter_deeplearning_df.sort_values('Tdelta')

                                        predict_df_s.to_csv('deeplearning/predict/'
                                                            + savename
                                                            + '_predict.csv')

                                        # evaluate by the for - loop     Not use now


                                        i=0
                                        predict_df = pd.DataFrame()
                                        output_delta_temp = 100000

                                        start_time = time.time()
                                        #print('start the for loop')
                                        for xpot_, ypot_, tend_, tside_, tmesh_, hter_ in itertools.product(xpot_candidate, ypot_candidate, tend_candidate, tside_candidate, tmesh_candidate, hter_candidate):

                                            input_ori = [xpot_, ypot_, tend_, tside_, tmesh_, hter_]

                                            xpot_ = xpot_ / xpot_coef
                                            ypot_ = ypot_ / ypot_coef
                                            tend_ = tend_ / tend_coef
                                            tside_ = tside_ / tside_coef
                                            tmesh_ = tmesh_ / tmesh_coef
                                            hter_ = hter_ / hter_coef

                                            input = [xpot_, ypot_, tend_, tside_, tmesh_, hter_]

                                            ##print(input)
                                            input = np.reshape(input, [1,6])

                                            output = eval_bydeeplearning(input)
                                            output = output * 1000
                                            output_max = float(max(output[0]))
                                            output_min = float(min(output[0]))
                                            ##print(output_max)
                                            ##print(output_min)
                                            output_delta = float(output_max - output_min)

                                            tmp_series = pd.Series([i,input_ori[0],input_ori[1],input_ori[2],input_ori[3],input_ori[4],input_ori[5],output[0][0],output[0][1],output[0][2],output[0][3],output_max,output_min,output_delta])

                                            if output_delta < output_delta_temp * 1.05:
                                                output_delta_temp = min(output_delta,output_delta_temp)

                                                predict_df = predict_df.append(tmp_series,ignore_index = True)

                                            i +=1
                                            #if i > 100 :
                                            #    break


                                        end_time = time.time()
                                        total_time = end_time - start_time
                                        #print('loop time is ', total_time)




                else:
                    units_size=62
                    bn_where=1
                    keep_prob=0.2
                    for ac_last in [1]:
                        model = get_model(layers_depth, units_size, bn_where, ac_last, keep_prob)

                        model.fit(train_input, train_output,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                validation_data=(test_input, test_output),
                                callbacks=[es_cb])

                        score_test = model.evaluate(test_input, test_output, verbose=0)
                        score_train = model.evaluate(train_input, train_output, verbose=0)
                        test_predict = model.predict(test_input, batch_size=32, verbose=1)
                        train_predict = model.predict(train_input, batch_size=32, verbose=1)

                        df_test_pre = pd.DataFrame(test_predict)
                        df_train_pre = pd.DataFrame(train_predict)

                        df_test_param = pd.DataFrame(test_output)
                        df_train_param = pd.DataFrame(train_output)

                        df_dens_test = pd.concat([df_test_param, df_test_pre], axis=1)
                        df_dens_train = pd.concat([df_train_param, df_train_pre], axis=1)

                        savename = ""
                        savename +=  '_score_train-' + str("%.3f" % round(math.log10(score_train[0]), 3))
                        savename +=  '_score_test-' + str("%.3f" % round(math.log10(score_test[0]), 3))
                        savename +=  '_numlayer-' + str(layers_depth)
                        savename +=  '_layersize-' + str(units_size)
                        savename +=  '_bn- ' + str(bn_where)
                        savename +=  '_ac-' + str(ac_last)
                        savename +=  '_k_p-' + str(keep_prob)
                        savename +=  '_patience-' + str(patience_)


                        df_dens_test.to_csv('deeplearning/traintest/' + savename + '_test.csv')

                        df_dens_train.to_csv('deeplearning/traintest/' + savename + '_train.csv')


                        model.save('deeplearning/h5/'
                                            + savename
                                            + '.h5')
                        # plot_model(model, to_file='C:\Deeplearning/model.png')
                        #plot_model(model, to_file='model.png')

                        #print('Test loss:', score_test[0])
                        #print('Test accuracy:', score_test[1])

                        ##print('predict', test_predict)

                        model.summary()
        '''





# settting
# fix the np.random.seed, it can get the same results every time to run this program
np.random.seed(1)
random.seed(1)


#########   regression by the scikitlearn model ###############
columns_results = ['model_name', 'train_model_mse', 'train_model_rmse', 'test_model_mse', 'test_model_rmse', 'train_model_score', 'test_model_score']
allmodel_results_df = pd.DataFrame(columns = columns_results)
allmodel_bayesian_opt_df = pd.DataFrame()

#print(allmodel_results_df)

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
t_csv = tkinter.StringVar()
entry_csv  = ttk.Entry(frame1, textvariable = t_csv, width = 40)

button_choose_csv    = ttk.Button(frame1, text='CSV選択',
                                 command = choose_csv, style = 'my.TButton')




label_theme_name  = tkinter.ttk.Label(frame1, text = 'テーマ名を入力:')
t_theme_name = tkinter.StringVar()
entry_theme_name  = ttk.Entry(frame1, textvariable = t_theme_name, width = 10)

label_id_clm_num   = tkinter.ttk.Label(frame1, text = '非データの列数を入力:')
label_input_clm_num   = tkinter.ttk.Label(frame1, text = '入力変数の列数を入力:')
label_output_clm_num   = tkinter.ttk.Label(frame1, text = '出力変数の列数を入力:')

t_info_clm_num = tkinter.StringVar()
t_input_clm_num = tkinter.StringVar()
t_output_clm_num = tkinter.StringVar()

entry_id_clm_num  = ttk.Entry(frame1, textvariable = t_info_clm_num, width = 10)
entry_input_clm_num  = ttk.Entry(frame1, textvariable = t_input_clm_num, width = 10)
entry_output_clm_num  = ttk.Entry(frame1, textvariable = t_output_clm_num, width = 10)

save_folder_name = os.path.dirname(t_csv.get()) + 'result'





Booleanvar_sklearn = tkinter.BooleanVar()
Booleanvar_deeplearning = tkinter.BooleanVar()
Booleanvar_gridsearch = tkinter.BooleanVar()
Booleanvar_bayesian_opt = tkinter.BooleanVar()

var_bayesian = tkinter.IntVar()
var_bayesian.set(0)


Booleanvar_sklearn.set(True)
Booleanvar_deeplearning.set(False)

Checkbutton_sklearn = tkinter.Checkbutton(frame1, text = '機械学習', variable = Booleanvar_sklearn)
#Checkbutton_sklearn.pack()
Checkbutton_deeplearning = tkinter.Checkbutton(frame1, text = 'ディープラーニング', variable = Booleanvar_deeplearning)
#Checkbutton_deeplearning.pack()

Checkbutton_gridsearch = tkinter.Checkbutton(frame1, text = '全探索', variable = Booleanvar_gridsearch)
#Checkbutton_gridsearch.pack()

Checkbutton_bayesian_opt = tkinter.Checkbutton(frame1, text = 'ベイズ最適化', variable = Booleanvar_bayesian_opt)
#Checkbutton_gridsearch.pack()

Radiobutton_bayesian_max = tkinter.Radiobutton(frame1, value = 0, text = '最大化', variable = var_bayesian)
#Checkbutton_gridsearch.pack()
Radiobutton_bayesian_min = tkinter.Radiobutton(frame1, value = 1, text = '最小化', variable = var_bayesian)
#Checkbutton_gridsearch.pack()
Radiobutton_bayesian_val = tkinter.Radiobutton(frame1, value = 2, text = '目的値', variable = var_bayesian)
#Checkbutton_gridsearch.pack()

t_bayesian_val = tkinter.StringVar()
entry_bayesian_val = ttk.Entry(frame1, textvariable = t_bayesian_val, width = 5)


button_learning     = ttk.Button(frame1, text='訓練開始',
                                 command = learning, style = 'my.TButton')


# set canvas information

frame2 = tkinter.Toplevel()
frame2.title('graph')
frame2.geometry('800x800')
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

# design

label_csv.grid(row=2,column=1,sticky=E)
entry_csv.grid(row=2,column=2,sticky=W)
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
Checkbutton_deeplearning.grid(row = 10, column = 2, stick = W)
Checkbutton_gridsearch.grid(row = 11, column = 2, stick = W)
Checkbutton_bayesian_opt.grid(row = 12, column = 2, stick = W)

Radiobutton_bayesian_max.grid(row = 12, column = 3)
Radiobutton_bayesian_min.grid(row = 12, column = 4)
Radiobutton_bayesian_val.grid(row = 12, column = 5)
entry_bayesian_val.grid(row = 13, column = 5)

button_learning.grid(row = 14 , column = 2, sticky = W)


for child in frame1.winfo_children():
    child.grid_configure(padx=5, pady=5)

root.mainloop()
