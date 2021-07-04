from tensorflow import keras
from graphviz import Digraph;
import keras;
from keras.models import Sequential;
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM;
import json

def drawModel(model, view=True, filename="network.gv", title=""):
    modelJSON = json.loads(model.to_json())
    layers = modelJSON['config']['layers']
    layersToDraw = {}
    for index, layer in enumerate(layers):
        if 'units' not in layer['config'] and index == 0:
            continue
        else:
            if 'units' in layer['config']:
                layersToDraw[index] = [int(layer['config']['units']), layer['class_name']]
            elif layer['class_name'] in ['Dropout']:
                layersToDraw[index] = [1, layer['class_name']]

    g = Digraph('g', filename=filename, format='png', node_attr={'fixedsize': 'true', 'width': '0.4', 'height': '0.4'})
    g.attr(bgcolor='#181f18')#backgroundColor
    #g.attr(rankdir='LR', nodesep='0.1', ratio="compress", sep='0.1', splines='curved')#Orientation landscape
    g.attr(rankdir='LR', ranksep='1', nodesep='0.1', ratio="compress", sep='0.1', size='5', splines='line')#Orientation landscape
    
    lastKeyItem = list(layersToDraw)[-1]    
    nodesDrawn = 0
    
    cellColor = "yellow";
    
    for index, value in layersToDraw.items():
        with g.subgraph(name="cluster_"+str(index)) as c:
            c.attr(label='process #1', style='filled', color='lightgrey')
            if lastKeyItem >= index + 1:#Last layer
                #print('index', index)                
                nextK = nextKey(layersToDraw, index)
                nodesDrawn = nodesDrawn + value[0]
                cellColor1, fontcolor1, style1, shape1 = getLayerStyle(value[1])
                cellColor2, fontcolor2, style2, shape2 = getLayerStyle(layersToDraw[nextK][1])
                for i in range(1, layersToDraw[nextK][0] + 1):
                    for j in range(1, value[0] + 1):                        
                        if index - 1 not in layersToDraw:
                            #print(str(j), str(nodesDrawn + i))                            
                            g.edge(str(j), str(nodesDrawn + i))                            
                            g.node(name=str(j), color=cellColor1, style=style1, fontcolor=fontcolor1, shape=shape1)
                            g.node(name=str(nodesDrawn + i), color=cellColor2, style=style2, fontcolor=fontcolor2, shape=shape2)
                        else:
                            #print(str(nodesDrawn - value[0] + j), str(nodesDrawn + i))
                            g.edge(str(nodesDrawn - value[0] + j), str(nodesDrawn + i));
                            g.node(name=str(nodesDrawn - value[0] + j), color=cellColor2, style=style2, fontcolor=fontcolor2, shape=shape2)
                            g.node(name=str(nodesDrawn + i), color=cellColor1, style=style1, fontcolor=fontcolor1, shape=shape1)
            else:#other layers
                break

    g.attr(arrowShape="none");
    g.edge_attr.update(arrowhead="none", color="white");
    if view == True:
        g.view();

def getLayerStyle(layerType):
    color = None
    style = 'filled'
    shape = 'circle'
    if layerType == 'Dense':
        color = '#FF7F50'
        return color, color, style, shape
    if layerType == 'Conv2D':
        color = '#40E0D0'
        return color, color, style, shape
    if layerType == 'MaxPooling2D':
        color = '#00FFFF'
        return color, color, style, shape
    if layerType == 'Dropout':
        color = '#FF0000'
        return color, color, style, shape
    if layerType == 'Flatten':
        color = '#DE3163'
        return color, color, style, shape
    if layerType == 'LSTM':
        color = '#00FF00'
        return color, color, style, shape
    else:
        color = '#FF00FF'
        return color, color, style, shape

def nextKey(dictionary, key):
    dict_keys = list(dictionary.keys())
    try:
        return dict_keys[dict_keys.index(key) + 1]
    except IndexError:
        #print('Item index does not exist')
        return -1
    