from flask import Flask, render_template, request, redirect, url_for, send_from_directory

import os
from os import listdir
from os.path import isfile, join

import cv2
from urllib.request import urlopen
import numpy as np
import pandas as pd

import keras
from keras.models import load_model
from keras import backend

from bokeh.models import FactorRange, Plot, LinearAxis, Grid
from bokeh.plotting import figure, show
from bokeh.embed import components
from bokeh.models import CustomJS, ColumnDataSource, CustomJSFilter, CDSView, HoverTool, Range1d, CheckboxGroup, CategoricalColorMapper
from bokeh.layouts import widgetbox, column, row, layout
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider, CheckboxGroup
from bokeh.transform import jitter
from bokeh.models import BasicTickFormatter, NumeralTickFormatter

# image dimensions
img_rows, img_cols, img_color = 130, 130, 1
input_shape = (img_rows, img_cols, img_color)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
print('uploads folder: {}'.format(UPLOAD_FOLDER))

# connect the app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# crop and convert ot grayscale
def process_image(file, verbose=False):
    img = cv2.imread(file)
    rows, cols, channels = img.shape
    if verbose: 
        print('  rows {}, cols {}, channels {}'.format(rows, cols, channels))
    img = img[0:rows, 0:rows]
    img = cv2.resize(img, (img_rows, img_cols), interpolation = cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def get_predict(example, model, threshold=0.5):
    y_pred = model.predict(example)
    y_pred[y_pred>=threshold] = 1
    y_pred[y_pred<threshold] = 0
    #y_pred = list(map(int, y_pred)) # convert to int
    return y_pred

# create keras input from image file
def create_example(image, verbose=False):
    x = np.array(image)
    if verbose:
        print('  the shape of X pre-reshape: {} {}'.format(x.shape))
    x = x.reshape(1, img_rows, img_cols, img_color)
    if verbose:
        print('  the shape of X post-reshape: {} {}'.format(x.shape))
    x = x.astype('float32')
    x /= 255.
    return x

def file_to_localfile(file, verbose=False):
    filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    if verbose:
        print('  the filename: {}, type: {}'.format(filename, type(filename)))
    file.save(filename)
    return filename

# load Keras models and run predict on the example
def extract(example, verbose=False):
    model_port = load_model('static/models/model_port.h5')
    portraits = model_port.predict(example)[0]
    if verbose:
        print('  portraits: {}, type: {}'.format(portraits, type(portraits)))
    # do this for some reason
    backend.clear_session()
    return portraits

# access the file, convert it to a NumPy array, read it
# into OpenCV format, and save it as a local image
def url_to_localfile(url, verbose=False):
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    name = url.split('/')[-1]
    filename = os.path.join(app.config['UPLOAD_FOLDER'], name)
    cv2.imwrite(filename, image)
    if verbose:
        print('  response: {}, type: {}'.format(resp, type(resp)))
        print('  image: {}, type: {}'.format(image, type(image)))
        print('  filename: {}, type: {}'.format(image, type(image)))
    return filename

def get_data(file, verbose=False):

    data = pd.read_csv(file, 
        delimiter=',', 
        encoding='utf-8-sig')

    data = data[~data['RIC'].isnull()]
    data['RIC'] = data['RIC'].apply(lambda x: x.split()[-1])
    data['RIC'] = data['RIC'].apply(lambda x: -1 if x=='-' or x=='–' else x)
    data['RIC'] = data['RIC'].astype(int)

    data['Grade'] = pd.Categorical(data['Grade'], 
        ordered=True,
        categories=['Fine', 
            'Good Fine', 
            'Near VF', 
            'VF', 
            'Good VF', 
            'Near EF', 
            'EF', 
            'Superb EF', 
            'Choice EF', 
            'FDC'])
    
    data['AuctionType'] = data['Auction Type']
    data['AuctionID'] = data['Auction ID']
    data['LotNumber'] = data['Lot Number'].apply(lambda x: x.split()[-1])
    data['LotNumber'] = data['LotNumber'].astype(int)

    if verbose:
        print(data.info())
        print(data['Grade'].value_counts())

    return data

# the plotting function
def make_plot(emperor, df, column=None, value=None):

    df = df[df['Emperor']==emperor]

    source = ColumnDataSource(df)

    select_den = Select(title="Denomination", 
        value="All", 
        options=["All", 
            "Aureus", 
            "Denarius"])

    select_grd = Select(title="Grade", 
        value="All", 
        options=['All', 
            'Fine', 
            'Good Fine', 
            'Near VF', 
            'VF', 
            'Good VF', 
            'Near EF', 
            'EF', 
            'Superb EF', 
            'Choice EF', 
            'FDC'])
    
    check_attractive = CheckboxGroup(labels=["Attractively Toned"], 
        active=[])

    check_cabinet = CheckboxGroup(labels=["Cabinet Toning"], 
       active=[])

    check_lusterous = CheckboxGroup(labels=["Lusterous"], 
       active=[])

    check_centered = CheckboxGroup(labels=["Well Centered"], 
       active=[])

    check_portrait = CheckboxGroup(labels=["Artistic Portrait"], 
       active=[])

    # This callback is crucial, otherwise the filter will 
    # not be triggered when the select changes
    callback = CustomJS(args=dict(source=source), code="""
        source.change.emit();
    """)

    select_den.js_on_change('value', callback)
    select_grd.js_on_change('value', callback)
    
    check_attractive.js_on_change('active', callback)
    check_cabinet.js_on_change('active', callback)
    check_lusterous.js_on_change('active', callback)
    check_centered.js_on_change('active', callback)
    check_portrait.js_on_change('active', callback)


    custom_filter_grd = CustomJSFilter(args=dict(select=select_grd, source=source), 
        code = '''
        var indices = [];
        console.log(select.value);
        // no select cuts applied
        if (select.value == 'All') {
            for (var i = 0; i < source.get_length(); i++){
                indices.push(true);
            }
            return indices;
        }
        // iterate through rows of data source and see if each satisfies some constraint
        for (var i = 0; i < source.get_length(); i++){
            if (source.data['Grade'][i] == select.value){
                indices.push(true);
            } else {
                indices.push(false)
            }
        }
        return indices;
    ''')

    custom_filter_den = CustomJSFilter(args=dict(select=select_den, source=source), 
        code = '''
        var indices = [];
        console.log(select.value);
        // no select cuts applied
        if (select.value == 'All') {
            for (var i = 0; i < source.get_length(); i++){
                indices.push(true);
            }
            return indices;
        }
        // iterate through rows of data source and see if each satisfies some constraint
        for (var i = 0; i < source.get_length(); i++){
            if (source.data['Denomination'][i] == select.value){
                indices.push(true);
            } else {
                indices.push(false)
            }
        }
        //console.log(indices)
        return indices;
    ''')

    custom_filter_attractive = CustomJSFilter(args=dict(checkbox=check_attractive, source=source), 
        code = '''
        var indices = [];
        //console.log(checkbox.active);
        if (checkbox.active.includes(0)) {
            //console.log('0 on')
            //console.log(checkbox.active.includes(0));
            for (var i = 0; i < source.get_length(); i++) {
                if (source.data['Attractively Toned'][i] == 1) {
                    indices.push(true);
                } else {
                    indices.push(false)
                }
            }
        } else {
            //console.log('0 off')
            for (var i = 0; i < source.get_length(); i++) {
                indices.push(true);
            }
        }
        return indices;
    ''')

    custom_filter_cabinet = CustomJSFilter(args=dict(checkbox=check_cabinet, source=source), 
        code = '''
        var indices = [];
        //console.log(checkbox.active);
        if (checkbox.active.includes(0)) {
            //console.log('0 on')
            //console.log(checkbox.active.includes(0));
            for (var i = 0; i < source.get_length(); i++) {
                if (source.data['Cabinet Toning'][i] == 1) {
                    indices.push(true);
                } else {
                    indices.push(false)
                }
            }
        } else {
            //console.log('0 off')
            for (var i = 0; i < source.get_length(); i++) {
                indices.push(true);
            }
        }
        return indices;
    ''')

    custom_filter_lusterous= CustomJSFilter(args=dict(checkbox=check_lusterous, source=source), 
        code = '''
        var indices = [];
        //console.log(checkbox.active);
        if (checkbox.active.includes(0)) {
            //console.log('0 on')
            //console.log(checkbox.active.includes(0));
            for (var i = 0; i < source.get_length(); i++) {
                if (source.data['Lusterous'][i] == 1) {
                    indices.push(true);
                } else {
                    indices.push(false)
                }
            }
        } else {
            //console.log('0 off')
            for (var i = 0; i < source.get_length(); i++) {
                indices.push(true);
            }
        }
        return indices;
    ''')

    custom_filter_centered = CustomJSFilter(args=dict(checkbox=check_centered, source=source), 
        code = '''
        var indices = [];
        //console.log(checkbox.active);
        if (checkbox.active.includes(0)) {
            //console.log('0 on')
            //console.log(checkbox.active.includes(0));
            for (var i = 0; i < source.get_length(); i++) {
                if (source.data['Cabinet Toning'][i] == 1) {
                    indices.push(true);
                } else {
                    indices.push(false)
                }
            }
        } else {
            //console.log('0 off')
            for (var i = 0; i < source.get_length(); i++) {
                indices.push(true);
            }
        }
        return indices;
    ''')

    custom_filter_portrait = CustomJSFilter(args=dict(checkbox=check_portrait, source=source), 
        code = '''
        var indices = [];
        //console.log(checkbox.active);
        if (checkbox.active.includes(0)) {
            //console.log('0 on')
            //console.log(checkbox.active.includes(0));
            for (var i = 0; i < source.get_length(); i++) {
                if (source.data['Quality Portrait'][i] == 1) {
                    indices.push(true);
                } else {
                    indices.push(false)
                }
            }
        } else {
            //console.log('0 off')
            for (var i = 0; i < source.get_length(); i++) {
                indices.push(true);
            }
        }
        return indices;
    ''')

    view = CDSView(source=source, 
        filters=[custom_filter_grd, 
            custom_filter_den, 
            custom_filter_attractive, 
            custom_filter_cabinet,
            custom_filter_lusterous,
            custom_filter_centered,
            custom_filter_portrait])

    TOOLS = ["pan, wheel_zoom, box_zoom, reset, save"]

    TOOLTIPS = [
        ("Auction", "@AuctionID"),
        ("Lot", "@LotNumber"),
        ("Emperor", "@Emperor"),
        ("RIC Number", "@RIC"),
        ("Estimate [USD]", "@Estimate"),
        ("Sale [USD]", "@Sale")
    ]

    plot = figure(title='CNG Auctions through 2019 for Coins of Emperor '+emperor,
        plot_width=500, 
        plot_height=300, 
        tools=TOOLS, 
        tooltips=TOOLTIPS,
        x_range=Range1d(start=20, 
            end=80000, 
            bounds=(None, None)),
        y_range=['Fine', 
            'Good Fine', 
            'Near VF', 
            'VF', 
            'Good VF', 
            'Near EF', 
            'EF', 
            'Superb EF', 
            'Choice EF', 
            'FDC'],
        x_axis_type='log'
        )

    color_mapper = CategoricalColorMapper(factors=['Aureus', 'Denarius'], 
        palette=['#FFD700', '#C0C0C0'])

    plot.circle(x='Sale', 
        y=jitter('Grade', 0.4, range=plot.y_range), 
        source=source, 
        view=view, 
        fill_alpha=0.8, 
        size=5, 
        legend='data', 
        line_color=None, 
        color={'field': 'Denomination', 'transform': color_mapper}, 
        hover_color='red')
    
    plot.xaxis.axis_label = "Sale Price (USD)"
    plot.yaxis.axis_label = "Grade"
    plot.xaxis.formatter = BasicTickFormatter(use_scientific=False)
    plot.xaxis.formatter = NumeralTickFormatter(format='$0,0')
    plot.legend.visible = False

    return row(plot, 
        widgetbox([select_den, 
            select_grd, 
            check_attractive, 
            check_cabinet,
            check_lusterous,
            check_centered,
            check_portrait]))


@app.route('/')
def homepage():
    return render_template('homepage.html')

@app.route('/analysis')
def analysispage():
    return render_template('analysis.html')

@app.route('/about')
def aboutpage():
    return render_template('about.html')

@app.route('/contact')
def contactpage():
    return render_template('contact.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results', methods=['POST'])
def resultspage():

    # get form submit data
    filename = None
    try:
        file = request.files['coin']
        print('  file: {}, type: {}'.format(file, type(file)))
        filename = file_to_localfile(file)
        print('  file: {}, type: {}'.format(filename, type(filename)))
    except:
        print('  No image uploaded')
        pass
    try:
        url = request.form['url']
        print('  url: {}, type: {}'.format(url, type(url)))
        # access url, convert to image, save as local file
        filename = url_to_localfile(url)
        print('  file: {}, type: {}'.format(filename, type(filename)))
    except:
        print('  No url supplied')
        pass

    # format the image and create the keras input
    image = process_image(filename)
    example = create_example(image)

    # run predict and convert results to percentages
    portraits = extract(example)
    portraits = [round(p*100,2) for p in portraits]

    emperors = ['Augustus', 'Tiberius', 'Nero', 'Vespasian', 'Domitian', 
                'Trajan', 'Hadrian', 'Antoninus Pius', 'Marcus Aurelius']
    results = dict(zip(emperors, portraits))
    sorted_results = sorted(results.items(), key=lambda x: -x[1])
    emperors, probs = zip(*sorted_results)

    # build the dataframe
    data = get_data('static/data/data.csv')

    # build the plot
    plot = make_plot(emperors[0], data)
    script, div = components(plot)

    return render_template("results.html", 
        div=div, 
        script=script, 
        results=sorted_results[:3], 
        emperors=emperors[:3], 
        probs=probs[:3], 
        filename=filename)


if __name__ == '__main__':
    app.run(port=5000, debug=True)



