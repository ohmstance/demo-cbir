import io
import os
import webbrowser
import numpy as np
from base64 import b64encode
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for

import cbir

app = Flask(__name__, template_folder="templates")
app.config.update(
    TEMPLATES_AUTO_RELOAD=True
)

exiting = False

@app.route("/", methods=['GET', 'POST'])
def main_app():
    # Home page
    if request.method == 'GET':
        return render_template("index.html", classes=cbir.list_classes())
    
    # POST request API for image search
    if request.method == 'POST':
        # Get image from form
        file = request.files['search_img']
        img = Image.open(file.stream)
        
        # Save image to memory and pass to template in base64 encoded PNG
        mem_file = io.BytesIO()
        img.save(mem_file, 'PNG')
        search_img = "data:image/png;base64," + b64encode(mem_file.getvalue()).decode('ascii')
        
        # Finds similar images
        img = np.array(img, dtype=bool)
        sim = cbir.get_similar(img)
        
        # Organizes output into [[filename, image, distance], ...] for html template
        retrieved = []
        for r in sim:
            filename = r['filename']
            distance = f"{r['distance']:.3f}"
            
            img = Image.fromarray(r['image'])
            mem_file = io.BytesIO()
            img.save(mem_file, 'PNG')
            img = "data:image/png;base64," + b64encode(mem_file.getvalue()).decode('ascii')
            
            retrieved.append((filename, img, distance))
        
        # Keyword args that will be passed to html template
        html_vars = {
            'classes': cbir.list_classes(),
            'search_img': search_img,
            'search_img_name': file.filename,
            'retrieved': retrieved
        }
        
        # If calculate precision recall is checked, do calculation according to class and add to dict
        if request.form.get('calculate_pr', False):
            in_class = request.form['select_class']
            precision, recall = cbir.evaluate_search(in_class, sim)
            plot = cbir.plot_pr_graph(in_class, sim)
            plot = "data:image/png;base64," + b64encode(plot.getvalue()).decode('ascii')
            html_vars.update({
                'precision': precision,
                'recall': recall,
                'plot': plot
            })
        
        # Renders html page
        return render_template("index.html", **html_vars)

# Updates features of images in directory into features.pickle
@app.route("/update", methods=['GET'])
def update_features():
    cbir.update_features()
    return redirect(url_for('main_app'))

# Shuts down server
@app.route("/shutdown", methods=['GET'])
def shutdown():
    global exiting
    exiting = True
    return "Done"

# Teardown is after completing request -- this one shuts down server
# There's no straight forward way to shut down a Flask server
@app.teardown_request
def teardown(exception):
    if exiting:
        os._exit(0)

if __name__ == "__main__":
    print("Open URL: http://localhost:8080/")
    webbrowser.open('http://localhost:8080')
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
