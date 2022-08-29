
#############################
#### just adding a line ####

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: MARKDOWN
# # notebook title

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE

import dataiku
from dataiku.customrecipe import *
import pandas as pd, numpy as np

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import io

import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt


### READ PLUGIN INPUTS ###

# Retrieve input and output dataset names
input_dataset_name = get_input_names_for_role('input_dataset')[0]
output_folder_name = get_output_names_for_role('main_output')[0]

# Retrieve mandatory user-defined parameters
plot_title = get_recipe_config().get('plot_title', "distribution of predictions by a given acceptable error")
    
prediction = get_recipe_config().get('prediction')
actual_values = get_recipe_config().get('actual')
type_of_plot = get_recipe_config().get('type_of_plot')

# Retrieve optional user-defined parameters
max_margin = get_recipe_config().get('max_margin')
unit = get_recipe_config().get('unit')

# Read input dataset as dataframe
input_dataset = dataiku.Dataset(input_dataset_name)
dff = input_dataset.get_dataframe()

### ERROR CHECKING OF USER INPUTS ###

# Check that x, y and z axis correspond to column names
if (prediction not in dff.columns) or (actual_values not in dff.columns):
    raise ValueError("input columns are {} {}".format(actual_values,prediction))
    raise KeyError("prediction and actual_values parameters must be columns in the input dataset.")
    
# Check that x, y, and z axis columns contain numeric values
if (not is_numeric_dtype(dff[prediction])) or (not is_numeric_dtype(dff[actual_values])):
    raise ValueError("input columns are {} {}".format(actual_values,prediction))
    raise ValueError("prediction and actual_values columns should only contain numeric values.")

# Check that the filter column is part of the dataframe (if defined)
# if type(max_margin) != int:
#     raise KeyError("max margin must be numeric. it;s type: {}".format(type(max_margin)))

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#### writing functions

### creating scale of margin of error

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
#### writing functions

### creating scale of margin of error

def get_max_error(df,col_pred,col_real):
    df['error'] = df[col_pred]-df[col_real]
    return int(np.floor(max(df['error'])))
    
def create_margin(minn=1,jumps=1,maxx=10,line=False):
    if line:
        out = list(np.arange(0.1,maxx+1.1,0.1))
    else:
        out = list(range(minn,maxx+1,jumps))
    return out

### function for ploting margin of error 

def margin_of_error_plot(dfin,ypred,ytest,margin,line=False,units = ''):
    '''input predictiions and real test values from model
        input list of margin of errors'''
#     df = pd.DataFrame(zip(dfin[ypred],dfin[ytest]),columns=['predictions','real'])
    df = dfin[[ypred,ytest]]
    df['error']= df[ypred]-df[ytest]
#     margin = [1,2,3,4,5]
    line_minus = []
    line_plus = []
    vals_list = []
    orig_frst = pd.DataFrame()
    indx = []
    for i in margin:
        minus = len(df[df['error']<-i])/len(df)*100
        plus = len(df[df['error']>i])/len(df)*100
        good = 100 - minus-plus
    #     print(good)
        cols = ['predicted_low','predicted_in_margin','predicted_high']
        vals = [minus,good,plus]
        vals_list.append(vals)
        indx.append(i)
        orig_frst = orig_frst.append(pd.DataFrame(dict(zip(cols,vals)),index=[i]))
        line_minus.append(vals[0])
        line_plus.append(vals[0]+vals[1])
    print('len list',len(vals_list))
    print('len df',len(orig_frst))
    print(orig_frst)
#     cols2 = cols.append('index')
    frst = pd.DataFrame(data = vals_list,columns = cols,index=indx)
    print(frst)
#     print(line_minus)
    if line == True:
        plt.figure(figsize= (8,4))
        plt.plot(margin,line_minus,color = "orange")
        plt.plot(margin,line_plus,color = "orange")
        plt.ylim(ymin=0,ymax = 100)
        plt.xlabel(f"margin of error (in {units})", fontsize=14)
        plt.ylabel("percent of predictions", fontsize=14)
        plt.suptitle("area of acceptable error margin",fontsize=20)
        plt.title("the bigger the area between the lines and the faster they grow apart - the better the model performed in the chosen x-axis range",fontsize=8)
    else:
        plt.figure(figsize= (12,4))
        ax = frst.plot(kind='bar', stacked=True,figsize=(20, 5))
        plt.xlabel(f"margin of error (in {units})", fontsize=14)
        plt.ylabel("percent of predictions", fontsize=14)
        ax.legend(bbox_to_anchor=(1.0, 1.0))
        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            ax.text(x+width/2, 
                    y+height/2, 
                    
                    '{:.0f} %'.format(height), 
                    horizontalalignment='center', 
                    verticalalignment='center')
        plt.title("Distributions predictions inside and outside of acceptable margin of error",fontsize=20)
    return df



# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
## call functions
if max_margin is not None:
    maxxx = max_margin
else:
    maxxx = get_max_error(dff,prediction,actual_values)

# prediction = 'prediction'
# actual_values = 'Sales'

if type_of_plot=='line':
    margin = create_margin(maxx = maxxx,line = True)
    margin_of_error_plot(dff,prediction,actual_values,margin=margin,line = True,units = unit)
else:
    margin = create_margin(maxx = maxxx)
    margin_of_error_plot(dff,prediction,actual_values,margin=margin,units = unit)


# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Save scatter plot to folder
# Save scatter plot to folder

folder_for_plots = dataiku.Folder(output_folder_name)
import io
picture = io.BytesIO()
plt.savefig(picture, format='png')
file_name= 'margin'
folder_for_plots.upload_stream(f"{file_name}-3D.png", picture.getvalue())
