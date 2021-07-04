"""
Author: Andres F Silva
Date: June 2021
Description: Create a dashboard with common visualizations and test for
better understanding of time series and LSTM models built with keras
"""

import ipywidgets as widgets
from ipywidgets import VBox, HBox, Label, Layout, Output, AppLayout, Tab, Image
import pandas as pd
import numpy as np
from bqplot import pyplot as plt
from IPython.display import display, HTML
import statsmodels.tsa.seasonal as tsa
import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.tsa.stattools import adfuller, kpss
import statsmodels.api as sm
import re
import warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import json
from drawModel import drawModel

warnings.filterwarnings("ignore")


#Global Variables
commonData = None
ctrls = pd.DataFrame(columns=['uuid', 'control', 'value'])
liveTrainingLoss = []



#First tab output 
output = Output()
outDescribe = Output()
plotOutput = Output()
plotNormalize = Output()

#Second tab output
plotSDOutput = Output()

#Third tab output
adfOutput = Output()
kpssOutput = Output()
plotTestOutput = Output()

#Fourth Tab
plotHistoryOutput = Output()
plotModelResultsOutput = Output()
outErrorMetrics = Output()
plotModelFuturePredictions = Output()
outputDraw = Output()
userModel = None

def dashboardTS(data, option="dashboard", model=None):
    """
    Principal function to return a dashboard or an applayout
    """
    global commonData 
    commonData = data
    if model is not None:
        global userModel
        userModel = model
    if option == "dashboard":
        ddData = widgets.Dropdown(name = "dropData", description="Year", options = uniqueSortedValuesPlusAll(np.array(data.index.year)))
        ddInterpolate = widgets.Dropdown(name = "ddInterp", description="Interpolate", options = ['None', 'linear','time','index','values','pad','nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline',
                                                                    'barycentric', 'polynomial', 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima','cubicspline',
                                                                    'from_derivatives'])
        ddData.observe(DropdownListener, names='value')
        ddInterpolate.observe(DropdownListener, names='value')    
        widgetsRegister('ddData', ddData)
        widgetsRegister('ddInterpolate', ddInterpolate)
        
        ddSDMethod = widgets.Dropdown(name = "ddSDMethod", description="SD Method", options = ['additive','multiplicative'])
        ddSDPeriod = widgets.Dropdown(name = "ddSDPeriod", description="SD Period", options = [1,4,12,52,365])

        auxRange = uniqueSortedValuesPlusAll(np.array(data.index.year))
        auxRange.remove('All')

        yearSlider = widgets.IntRangeSlider(
            value=[auxRange[0],auxRange[-1]],
            min=auxRange[0],
            max=auxRange[-1],
            step=1,
            description='SD Range',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )

        #Observers
        ddSDMethod.observe(seasonalDecomposeListener, names='value')
        ddSDPeriod.observe(seasonalDecomposeListener, names='value')
        yearSlider.observe(seasonalDecomposeListener, 'value')

        #Register controls for dashboard
        widgetsRegister('ddSDMethod', ddSDMethod)
        widgetsRegister('ddSDPeriod', ddSDPeriod)
        widgetsRegister('yearSlider', yearSlider)

        refreshSecond = widgets.Button(description="Refresh")
        refreshSecond.on_click(refreshSecondTab)

        #List of controls
        ddTestADFautolag = widgets.Dropdown(name = "ddTestADFautolag", description="ADF autolag", options = ["AIC", "BIC", "t-stat"])
        ddTestKPSSReg = widgets.Dropdown(name = "ddTestKPSSReg", description="KPSS Regression", options = ["c", "ct"])
        ddTestKPSSNlags = widgets.Dropdown(name = "ddTestKPSSNlags", description="KPSS nlags", options = ["auto","legacy"])

        #Observers
        ddTestADFautolag.observe(TestListener, names='value')
        ddTestKPSSReg.observe(TestListener, names='value')
        ddTestKPSSNlags.observe(TestListener, names='value')

        #Register controls for dashboard
        widgetsRegister('ddTestADFautolag', ddTestADFautolag)
        widgetsRegister('ddTestKPSSReg', ddTestKPSSReg)
        widgetsRegister('ddTestKPSSNlags', ddTestKPSSNlags)

        #Refresh tab
        refreshThird = widgets.Button(description="Refresh")
        refreshThird.on_click(refreshThirdTab)

        #List of controls Model
        trainSlider = customIntSlider(0, 100, 70, 'Train %')
        lookBackSlider = widgets.IntSlider(value=1,min=0,max=20,description='lookback')
        ddCompileLoss = widgets.Dropdown(name = "ddCompileLoss", description="loss", options = ['mean_absolute_error'])
        ddCompileOptimizer = widgets.Dropdown(name = "ddCompileOptimizer", description="Optimizer", options = ['adam'])
        
        #Observers
        ddCompileLoss.observe(controlsListenerModel, names='value')
        ddCompileOptimizer.observe(controlsListenerModel, names='value')
        trainSlider.observe(controlsListenerModel, 'value')
        lookBackSlider.observe(controlsListenerModel, 'value')

        #Register controls Model
        widgetsRegister('trainSlider', trainSlider)
        widgetsRegister('lookBackSlider', lookBackSlider)
        widgetsRegister('ddCompileLoss', ddCompileLoss)
        widgetsRegister('ddCompileOptimizer', ddCompileOptimizer)

        #Refresh tab Model
        refreshFourth = widgets.Button(description="Fit model")
        refreshFourth.on_click(refreshFourthTab)

        tvt = VBox([Label(value="Train - Test Configuration Size"),trainSlider,lookBackSlider])
        mc = VBox([Label(value="Model Compile"),ddCompileLoss,ddCompileOptimizer,refreshFourth])
        head = HBox([tvt, mc])

        #Config for tensorflow
        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass

            

        FirstTab()
        SecondTab()
        ThirdTab()

        tabControl = Tab()

        tabFirst = AppLayout(header=HBox([ddData, ddInterpolate]),
                left_sidebar=None,
                center=VBox([plotOutput, plotNormalize]),
                right_sidebar=VBox([outDescribe, output]),
                pane_widths=[3, 3, 1],
                pane_heights=[1, 10, 1])
        
        tabSecond = AppLayout(header=None,
                left_sidebar=None,
                center=VBox([HBox([refreshSecond, yearSlider, ddSDMethod, ddSDPeriod], layout=Layout(height='auto', width='auto')), plotSDOutput]),
                right_sidebar=None,
                pane_widths=[3, 3, 1],)

        tabThird = AppLayout(header=None,
                left_sidebar=VBox([refreshThird, Label(value='------------  ADF  ---------------'), ddTestADFautolag, adfOutput, Label(value='-----------  KPSS  -------------'), ddTestKPSSReg, ddTestKPSSNlags, kpssOutput]),
                center=plotTestOutput,
                right_sidebar=None,
                pane_widths=[3, 3, 1],)

        tabFourth = AppLayout(header=head,
          left_sidebar=VBox([plotHistoryOutput,outErrorMetrics]),
          center=VBox([plotModelResultsOutput, outputDraw]),
          right_sidebar=None,
          pane_heights=[1, 10, 1],
          pane_widths=[2, 10, 0])
        


        #TabBar
        tabControl.set_title(0, 'Data')
        tabControl.set_title(1, 'Seasonal decompose')
        tabControl.set_title(2, 'Tests')
        tabControl.set_title(3, 'Model')
        tabControl.children = [tabFirst,tabSecond, tabThird, tabFourth]

        return tabControl




def lenzi(df):
    """Check if a pandas series is empty"""
    return len(df.index) == 0

def getEnumerationFromArray(arr):
    """Return the enumeration of an array"""
    return list(map(lambda c: c[0], enumerate(arr)))

def getOptionsControl(control):
    """Return the options of DropDownControl"""
    return re.search("description=\'([^']+)\'", str(control)).group(0)

def widgetsRegister(uuid, control):
    """Register a control for reuse of functions"""
    global ctrls
    if lenzi(ctrls.loc[ctrls['uuid'].isin([uuid])]):
        key = getOptionsControl(control)
        newControl = pd.Series(data={'uuid':uuid, 'control':key, 'value':control.value}, index=['uuid', 'control', 'value'])
        ctrls = ctrls.append(newControl, ignore_index=True)

def updateRegisterValue(control, value):
    """Update value of control in the register"""
    global ctrls
    ctrls.at[ctrls.index[(ctrls['control'] == control)].tolist()[0], 'value'] = value


#Study if necesary or upgrade
def uniqueSortedValuesPlusAll(listOfValues):
    unique = np.unique(listOfValues).tolist()
    unique.sort()
    unique.insert(0, 'All')
    return unique

def changeModelInput(modelJSON, inputShape):
    modelJSON = json.loads(modelJSON)
    previousInputShape = None
    if modelJSON['config']['layers'][0]['class_name'] == 'InputLayer':
        previousInputShape = modelJSON['config']['layers'][0]['config']['batch_input_shape']
        modelJSON['config']['layers'][0]['config']['batch_input_shape'] = inputShape
    if modelJSON['config']['layers'][1]['class_name'] == 'LSTM' and modelJSON['config']['layers'][0]['config']['batch_input_shape'] == previousInputShape:
        modelJSON['config']['layers'][0]['config']['batch_input_shape'] = inputShape
    return json.dumps(modelJSON)

def FirstTab():
    """Get the values of controls and process"""
    #Read global controls registered
    global ctrls, commonData, normData

    data = commonData

    #Clear de previous output for dataframe or series - plot
    output.clear_output()
    outDescribe.clear_output()
    plotOutput.clear_output()
    plotNormalize.clear_output()

    #Load dashboard features
    ddDataFilter = ctrls.loc[ctrls['uuid'].isin(['ddData'])]['value']
    ddDataInterpolate = ctrls.loc[ctrls['uuid'].isin(['ddInterpolate'])]['value']
    
    #Check if param exist
    if not lenzi(ddDataFilter):
        #Check interpolate control
        if not lenzi(ddDataInterpolate):
            if ddDataInterpolate[1] == 'None':
                commonFilter = data
            else:
                commonFilter = data.interpolate(method = ddDataInterpolate[1])
        else:
            if ddDataFilter[0] == 'All':
                commonFilter = data
            else:
                commonFilter = data[data.index.year == ddDataFilter[0]]
    commonData = commonFilter
    normalizeData()
    Draw(commonFilter, normData)

def Draw(df, normData):
    """Draw the widgets"""
    #Display for dataframe or series with ipywidgets
    with output:
        display(df)

    #Display description for dataframe data
    with outDescribe:
        nanVal = df.isna().any()
        nanVal.name = 'Null Values'        
        display(df.describe().append(nanVal).round(2))

    #Display de plot of the data
    with plotOutput:
        scat_fig = plt.figure(title='Historia precio de cierre', animation_duration=100, figsize=(16,8))
        plt.title(df.columns[0])
        plt.plot(x=df.index, y=df[df.columns[0]])
        plt.xlabel('Date', fontsize=18)
        plt.show()

    with plotNormalize:
        fig = plt.figure(title='Historia precio de cierre', animation_duration=100, figsize=(16,8))
        plt.title(df.columns[0])
        plt.plot(x=df.index, y=normData)
        plt.xlabel('Date', fontsize=18)
        plt.show()

#**************************************************************** Second Tab **********************************************************************
def SecondTab():
    """Get the values of controls and process"""
    #Read global controls registered
    global ctrls, commonData
    #Clear de previous output for dataframe or series - plot
    plotSDOutput.clear_output()
    commonFilter = None
    result = None
    
    #Load dashboard features
    ddSDMethod = ctrls.loc[ctrls['uuid'].isin(['ddSDMethod'])]['value']
    ddSDPeriod = ctrls.loc[ctrls['uuid'].isin(['ddSDPeriod'])]['value']
    yearSlider = ctrls.loc[ctrls['uuid'].isin(['yearSlider'])]['value']
    
    #Check if param exist
    if not lenzi(yearSlider):        
        commonFilter = commonData[str(yearSlider[4][0]):str(yearSlider[4][1])].copy()
        #Check method and period controls
        try:
            result = tsa.seasonal_decompose(commonFilter, model=ddSDMethod[2], period=ddSDPeriod[3])
        except ValueError as error:
            result = str(error)
        
    drawSeasonalDecompose(result)

def drawSeasonalDecompose(df):
    """Draw the widgets for seasonal decompose"""
    #Display de plot of the data    
    if not isinstance(df, str):
        with plotSDOutput:
            fig_layout = widgets.Layout(width='auto', height='auto')
            fig_marg = dict(top=30, bottom=20, left=50, right=20)
            line_style = {'colors': ['magenta'], 'stroke_width': 3}

            figure1 = plt.figure(title='Observed', animation_duration=100, layout=fig_layout, fig_margin = fig_marg)
            plt.title('Observed')
            plt.plot(x=df.observed.index, y=df.observed, colors=['Green'])

            figure2 = plt.figure(title='Trend', animation_duration=100, layout=fig_layout, fig_margin = fig_marg)
            plt.title('Trend')
            plt.plot(x=df.trend.index, y=df.trend, colors=['Olive'])

            figure3 = plt.figure(title='Seasonal', animation_duration=100, layout=fig_layout, fig_margin = fig_marg)
            plt.title('Seasonal')
            plt.plot(x=df.seasonal.index, y=df.seasonal, colors=['SlateGray'])

            figure4 = plt.figure(title='Residual', animation_duration=100, layout=fig_layout, fig_margin = fig_marg)
            plt.title('Residual')
            plt.plot(x=df.resid.index, y=df.resid, colors=['DarkSlateGray'])

            display(VBox([figure1,figure2,figure3,figure4], layout=Layout(align_content='stretch')))
    else:
        with plotSDOutput:
            display(widgets.Label(value=df))

#**************************************************************** Third tab *********************************************
def ThirdTab():
    """Get the values of controls and process"""
    #Read global controls registered
    global ctrls, commonData
    #Clear de previous output for dataframe or series - plot
    adfOutput.clear_output()
    kpssOutput.clear_output()
    plotTestOutput.clear_output()
    adfTest = None
    kpssTest = None
    
    #Load dashboard features
    ddTestADFautolag = ctrls.loc[ctrls['uuid'].isin(['ddTestADFautolag'])]['value']
    ddTestKPSSReg = ctrls.loc[ctrls['uuid'].isin(['ddTestKPSSReg'])]['value']
    ddTestKPSSNlags = ctrls.loc[ctrls['uuid'].isin(['ddTestKPSSNlags'])]['value']
    
    #Check if param exist
    if not lenzi(ddTestADFautolag):
        #Check method and period controls
        try:
            #Calculate the adfuller
            adfTest = adfuller(commonData, autolag=str(ddTestADFautolag[5]))
            dfoutput = pd.Series(adfTest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
            for key,value in adfTest[4].items():
                dfoutput['Critical Value (%s)'%key] = value
        except Exception as error:
            dfoutput = str(error)
            
    #Check if param exist
    if not lenzi(ddTestKPSSReg) and not lenzi(ddTestKPSSNlags):
        #Check method and period controls
        try:
            #Calculate the kpss
            kpssTest = kpss(commonData, regression=str(ddTestKPSSReg[6]), nlags=str(ddTestKPSSNlags[7]))
            kpss_output = pd.Series(kpssTest[0:3], index=['Test Statistic','p-value','Lags Used'])
            for key,value in kpssTest[3].items():
                kpss_output['Critical Value (%s)'%key] = value            
        except Exception as error:
            kpss_output = str(error)
        
    drawTests(dfoutput, kpss_output)

def drawTests(adfTest, kpssTest):
    """Draw the widgets for seasonal decompose"""
    #Display test pacf, acf for dataframe data
    with plotTestOutput:
            fig_layout = widgets.Layout(width='auto', height='auto')
            fig_marg = dict(top=30, bottom=20, left=50, right=20)

            figure5 = plt.figure(title='Autocorrelation', animation_duration=100, layout=fig_layout, fig_margin = fig_marg)            
            acf, ci = sm.tsa.acf(commonData, alpha=0.05)
            x = getEnumerationFromArray(acf)
            plt.scatter(x=x, y=acf, marker='circle', colors=['Green'])


            figure6 = plt.figure(title='Partial Autocorrelation', animation_duration=100, layout=fig_layout, fig_margin = fig_marg)
            pacf, ci = sm.tsa.pacf(commonData, alpha=0.05)
            x = getEnumerationFromArray(pacf)
            plt.scatter(x=x, y=pacf, marker='circle', colors=['Green'])

            #figure3 = plt.figure(title='Seasonal', animation_duration=100, layout=fig_layout, fig_margin = fig_marg)
            #plt.title('Seasonal')
            #plt.plot(x=df.seasonal.index, y=df.seasonal, colors=['SlateGray'])

            #figure4 = plt.figure(title='Residual', animation_duration=100, layout=fig_layout, fig_margin = fig_marg)
            #plt.title('Residual')
            #plt.plot(x=df.resid.index, y=df.resid, colors=['DarkSlateGray'])

            display(VBox([figure5,figure6], layout=Layout(align_content='stretch')))

    #Display test 
    with adfOutput:
        display(adfTest)
        
    with kpssOutput:
        display(kpssTest)

#*********************************************************** FourthTab
def fourthTab(modelProvided=None):
    """Get the values of controls and process"""
    #Read global controls registered
    global ctrls, commonData, normData, liveTrainingLoss
    history = None
    #Clear de previous output for dataframe or series - plot
    
    plotHistoryOutput.clear_output()
    plotModelResultsOutput.clear_output()
    outErrorMetrics.clear_output()
    outputDraw.clear_output()

    #Load dashboard features
    trainSlider = ctrls.loc[ctrls['uuid'].isin(['trainSlider'])]['value']
    #validSlider = ctrls.loc[ctrls['uuid'].isin(['validSlider'])]['value']
    #testSlider = ctrls.loc[ctrls['uuid'].isin(['testSlider'])]['value']
    
    lookBackSlider = ctrls.loc[ctrls['uuid'].isin(['lookBackSlider'])]['value']
    ddCompileLoss = ctrls.loc[ctrls['uuid'].isin(['ddCompileLoss'])]['value']
    ddCompileOptimizer = ctrls.loc[ctrls['uuid'].isin(['ddCompileOptimizer'])]['value']
    
    trainPredict = None
    testPredict = None
    
    #Check if param exist
    if not lenzi(trainSlider) and not lenzi(ddCompileLoss) and not lenzi(ddCompileOptimizer):
        
        trainSize = int(len(normData)*(trainSlider[8]/100))
        testSize =  len(normData) - trainSize
        
        lookB = lookBackSlider[9]
        lossMethod = ddCompileLoss[10]
        
        train, test = normData[:trainSize], normData[trainSize:]        
        
        trainX, trainY = create_dataset(train, lookB)
        testX, testY = create_dataset(test, lookB)
        
        #Change form to: [samples, time_steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        
        model = None
        if modelProvided != None:
            newInputShape = [None, 1, lookB]
            model = keras.models.model_from_json(changeModelInput(modelProvided.to_json(), newInputShape))
        else:
            # Red LSTM
            model = keras.Sequential()
            model.add(LSTM(10, input_shape=(1, lookB)))
            model.add(Dense(1))
            
        model.compile(loss=lossMethod, optimizer=ddCompileOptimizer[11])
        liveTrainingLoss = []
        with plotHistoryOutput:
            display(HBox([Label(value="Entrenando modelo")]))
        with outputDraw:
            display(HBox([Label(value="Dibujando la red")]))
        drawModel(model)
        history = model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2, callbacks=[myCallback])
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        
        trainPredict, trainY = normalizeData(inverseTransform=True, pred=trainPredict, arr=trainY)
        testPredict, testY = normalizeData(inverseTransform=True, pred=testPredict, arr=testY)
        
        calculateErrors(trainPredict, trainY, testPredict, testY, lossMethod)
        
    drawModelHistory(history, live=False)
    drawModelResults(commonData, trainPredict, testPredict, lookB, commonScaler)
    getDrawingModel(model)

def drawModelHistory(history, live=True):
    plotHistoryOutput.clear_output()
    with plotHistoryOutput:
        figures = []
        if live:
            #for item in history.items():
            figures.append(customPlot('live', getEnumerationFromArray(history), history))
        else:
            for item in history.history:
                figures.append(customPlot(item, getEnumerationFromArray(history.history[item]), history.history[item]))            
        display(VBox(figures, layout=Layout(align_content='stretch')))
        
        
def customPlot(title, x, y):
    fig_layout = widgets.Layout(width='auto', height='auto')
    fig_marg = dict(top=30, bottom=20, left=50, right=20)
    line_style = {'colors': ['magenta'], 'stroke_width': 3}
    figure = plt.figure(title=title, layout=fig_layout, fig_margin = fig_marg)    
    plt.plot(x=x, y=y, colors=['Green'])
    figure.axes[1].num_ticks = 3
    return figure
        
def drawModelResults(data, trainPredict, testPredict, lookBack, commonScaler):
    with plotModelResultsOutput:
        trainPredictPlot = np.empty_like(data)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[lookBack:len(trainPredict)+lookBack, :] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(data)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict)+(lookBack*2)+1:len(data)-1, :] = testPredict

        figure = plt.figure(title='Real VS Predictions', animation_duration=100, figsize=(16,8))
        plt.plot(x=data.index, y=data[data.columns[0]])
        plt.plot(x=data.index, y=trainPredictPlot, colors=['Orange'])
        plt.plot(x=data.index, y=testPredictPlot, colors=['Green'])
        plt.xlabel('Date', fontsize=18)
        plt.show()
        figure.axes[1].num_ticks = 3
        #display(VBox([figure], layout=Layout(align_content='stretch')))
    with plotModelFuturePredictions:
        figure = plt.figure(title='Real VS Predictions', animation_duration=100, figsize=(16,8))

def getDrawingModel(model):
    outputDraw.clear_output()
    with outputDraw:
        file = open("network.gv.png", "rb")
        image = file.read()        
        display(Image(value=image, format='png', width=400, height=200,))

#Listeners and controls
#First Tab
def DropdownListener(change):
    """
    control Listener for fisrt tab 
    """
    result = getOptionsControl(change.owner)
    updateRegisterValue(result, change.new)
    FirstTab()

#Second tab
def seasonalDecomposeListener(change):
    """
    control Listener for Second tab 
    """
    result = getOptionsControl(change.owner)
    updateRegisterValue(result, change.new)
    SecondTab()

#Third tab
def TestListener(change):
    """
    control Listener for Second tab 
    """
    result = getOptionsControl(change.owner)
    updateRegisterValue(result, change.new)
    ThirdTab()

#Fourth tab
def controlsListenerModel(change):
    result = getOptionsControl(change.owner)
    updateRegisterValue(result, change.new)


#Refresh tab
def refreshSecondTab(b):
    SecondTab()

def refreshThirdTab(b):
    ThirdTab()

def refreshFourthTab(b):
    global userModel
    fourthTab(userModel)

#********************************************** model custom functions 
class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        plotHistoryOutput.clear_output()
        with plotHistoryOutput:
            global liveTrainingLoss
            liveTrainingLoss.append(logs['loss'])
            drawModelHistory(liveTrainingLoss)
        
myCallback = MyCallback()


def create_dataset(dataset, h=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-h-1):
        a = dataset[i:(i+h)]
        dataX.append(a)
        dataY.append(dataset[i + h])
    return np.array(dataX), np.array(dataY)

def normalizeData(inverseTransform=False, pred=None, arr=None):
    """
    Normalize the data provided
    """
    global commonData, normData, commonScaler    
    if not inverseTransform:        
        commonScaler = MinMaxScaler(feature_range=(0, 1))
        normData = commonData.values
        normData = normData.astype('float32')
        normData = normData.reshape(-1, 1)
        #print(normData)
        normData = commonScaler.fit_transform(normData)
    else:
        #print(pred)
        xInverse = commonScaler.inverse_transform(pred)
        yInverse = commonScaler.inverse_transform(arr)
        return xInverse, yInverse
    
def calculateErrors(trainPredict, trainY, testPredict, testY, measure):
    with outErrorMetrics:
        trainScore = None
        testScore = None
        if measure == 'mean_squared_error':
            print("ENTRA")
            trainScore = mean_squared_error(trainY, trainPredict)
            trainScore = ('Train Score: %.2f RMSE' % (trainScore))
            testScore = mean_squared_error(testY, testPredict)
            testScore = ('Test Score: %.2f RMSE' % (testScore))
        elif measure == 'mean_absolute_error':
            trainScore = mean_absolute_error(trainY, trainPredict)
            trainScore = ('Train Score: %.2f MAE' % (trainScore))
            testScore = mean_absolute_error(testY, testPredict)
            testScore = ('Test Score: %.2f MAE' % (testScore))
        #TODO:Implement more metrics mehtod
        display(VBox([Label(value=trainScore),Label(value=testScore)]))



#***************************************************** custom controls
def customIntSlider(minValue, maxValue, value, nameControl):
    return widgets.IntSlider(value=value,
                                min=minValue,
                                max=maxValue,
                                step=1,
                                description=nameControl,
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='d')
