import torch

import plotly
import numpy as np
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Adapted from https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5

def display_pca_scatterplot(p, model, word_vectors = None, dim=3, pca = True, perplexity = 0, learning_rate = 0, iteration = 0, 
                            title = None, show = True, return_data = False):

    assert dim in [2,3]
    #if model.hparams.regression : return
    device = model.device
    words = torch.arange(p).to(device)
    if word_vectors is None : w = model.embeddings(words).detach().cpu()
    else : w = word_vectors.detach().cpu() + 0
    word_vectors = w.numpy()
    words = words.cpu().tolist()
    if w.size(1) > dim and w.size(0) != 1:
        if pca :
            data_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:dim]
        else :
            data_dim = TSNE(n_components = dim, random_state=0, perplexity = perplexity, learning_rate = learning_rate, 
                            n_iter = iteration).fit_transform(word_vectors)[:,:dim]
    else :
        dim = w.size(0)
        data_dim = word_vectors
        if w.size(0) == 1 :
            p = w.size(1)
            words = torch.arange(p).tolist()
            dim = 2
    if return_data : return data_dim, words
    data = []
    for i in range(p):
        if dim==3 :
            trace = go.Scatter3d(
                x = data_dim[i:i+1,0], 
                y = data_dim[i:i+1,1],  
                z = data_dim[i:i+1,2],
                text = words[i],
                name = str(words[i]),
                textposition = "top center",
                textfont_size = 20,
                mode = 'markers+text',
                marker = {'size': 10, 'opacity': 0.8, 'color': 2}
            )
        else :
            trace = go.Scatter(
            #trace = go.Contour(
                x = data_dim[i:i+1,0], 
                y = data_dim[i:i+1,1],  
                text = words[i],
                name = str(words[i]),
                textposition = "top center",
                textfont_size = 20,
                mode = 'markers+text',
                marker = {'size': 10, 'opacity': 0.8, 'color': 2}
            )
                            
        data.append(trace)

    if dim==3 :
        trace_input = go.Scatter3d(
            x = data_dim[p:,0], 
            y = data_dim[p:,1],  
            z = data_dim[p:,2],
            text = words[p:],
            name = 'input words',
            textposition = "top center",
            textfont_size = 20,
            mode = 'markers+text',
            marker = {'size': 10, 'opacity': 1, 'color': 'black'},
            #title="Manually Specified Labels"
          )
    else :
        trace_input = go.Scatter(
        #trace_input = go.Contour(
            x = data_dim[p:,0], 
            y = data_dim[p:,1],  
            text = words[p:],
            name = 'input words',
            textposition = "top center",
            textfont_size = 20,
            mode = 'markers+text',
            marker = {'size': 10, 'opacity': 1, 'color': 'black'},
            #title="Manually Specified Labels"
          )
            
    data.append(trace_input)
    
    # Configure the layout
    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
    
        #showlegend=True,
        showlegend=False,
        
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        #width = 1000,
        #height = 1000,
        width = 500,
        height = 400
        )

    plot_figure = go.Figure(data = data, layout = layout)

    if title is not None :
        plot_figure.update_layout(
            #title=title,
            xaxis_title=title,
            #yaxis_title="Y Axis Title",
            #legend_title="Legend Title",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )

        # plot_figure.update_layout(
        #   title={
        #     'text': title,
        #     'y':0.9,
        #     'x':0.5,
        #     #'xanchor': 'center',
        #     #'yanchor': 'top'
        #     }
        #   )

    if show : plot_figure.show()

    return plot_figure


# Adapted from /usr/local/lib/python3.7/dist-packages/plotly/io/_kaleido.py

"""
try :
    from kaleido.scopes.plotly import PlotlyScope
except ModuleNotFoundError :
    ! pip install -U kaleido
"""

#! pip install -U kaleido

from six import string_types
import os
import json
from pathlib import Path
import plotly
from plotly.io._utils import validate_coerce_fig_to_dict

try:
    from kaleido.scopes.plotly import PlotlyScope

    scope = PlotlyScope()

    # Compute absolute path to the 'plotly/package_data/' directory
    root_dir = os.path.dirname(os.path.abspath(plotly.__file__))
    package_dir = os.path.join(root_dir, "package_data")
    scope.plotlyjs = os.path.join(package_dir, "plotly.min.js")
    if scope.mathjax is None:
        scope.mathjax = (
            "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js"
        )
except ImportError:
    PlotlyScope = None
    scope = None
    raise ValueError(
                """
    Image export using the "kaleido" engine requires the kaleido package,
    which can be installed using pip:
        $ pip install -U kaleido
    """)

def to_image(
    fig, format=None, width=None, height=None, scale=None, validate=True
):
    """
    Convert a figure to a static image bytes string

    Parameters
    ----------
    fig:
        Figure object or dict representing a figure

    format: str or None
        The desired image format. One of
          - 'png'
          - 'jpg' or 'jpeg'
          - 'webp'
          - 'svg'
          - 'pdf'
          - 'eps' (Requires the poppler library to be installed and on the PATH)

        If not specified, will default to:
             - `plotly.io.kaleido.scope.default_format` if engine is "kaleido"
             - `plotly.io.orca.config.default_format` if engine is "orca"

    width: int or None
        The width of the exported image in layout pixels. If the `scale`
        property is 1.0, this will also be the width of the exported image
        in physical pixels.

        If not specified, will default to:
             - `plotly.io.kaleido.scope.default_width` if engine is "kaleido"
             - `plotly.io.orca.config.default_width` if engine is "orca"

    height: int or None
        The height of the exported image in layout pixels. If the `scale`
        property is 1.0, this will also be the height of the exported image
        in physical pixels.

        If not specified, will default to:
             - `plotly.io.kaleido.scope.default_height` if engine is "kaleido"
             - `plotly.io.orca.config.default_height` if engine is "orca"

    scale: int or float or None
        The scale factor to use when exporting the figure. A scale factor
        larger than 1.0 will increase the image resolution with respect
        to the figure's layout pixel dimensions. Whereas as scale factor of
        less than 1.0 will decrease the image resolution.

        If not specified, will default to:
             - `plotly.io.kaleido.scope.default_scale` if engine is "kaleido"
             - `plotly.io.orca.config.default_scale` if engine is "orca"


    validate: bool
        True if the figure should be validated before being converted to
        an image, False otherwise.

    Returns
    -------
    bytes
        The image data
    """
    fig_dict = validate_coerce_fig_to_dict(fig, validate)
    img_bytes = scope.transform(
        fig_dict, format=format, width=width, height=height, scale=scale
    )

    return img_bytes


def write_image(
    file,
    image_bytes = None,
    fig = None,
    format=None,
    scale=None,
    width=None,
    height=None,
    validate=True
):
    """
    Convert a figure to a static image and write it to a file or writeable object

    Parameters
    ----------
    file: str or writeable
        A string representing a local file path or a writeable object
        (e.g. a pathlib.Path object or an open file descriptor)  
    fig:
        Figure object or dict representing a figure
    image_bytes :
        Image bytes
    format: str or None
        The desired image format. One of
          - 'png'
          - 'jpg' or 'jpeg'
          - 'webp'
          - 'svg'
          - 'pdf'
          - 'eps' (Requires the poppler library to be installed and on the PATH)

        If not specified and `file` is a string then this will default to the
        file extension. If not specified and `file` is not a string then this
        will default to:
            - `plotly.io.kaleido.scope.default_format` if engine is "kaleido"
            - `plotly.io.orca.config.default_format` if engine is "orca"

    width: int or None
        The width of the exported image in layout pixels. If the `scale`
        property is 1.0, this will also be the width of the exported image
        in physical pixels.

        If not specified, will default to:
            - `plotly.io.kaleido.scope.default_width` if engine is "kaleido"
            - `plotly.io.orca.config.default_width` if engine is "orca"

    height: int or None
        The height of the exported image in layout pixels. If the `scale`
        property is 1.0, this will also be the height of the exported image
        in physical pixels.

        If not specified, will default to:
            - `plotly.io.kaleido.scope.default_height` if engine is "kaleido"
            - `plotly.io.orca.config.default_height` if engine is "orca"

    scale: int or float or None
        The scale factor to use when exporting the figure. A scale factor
        larger than 1.0 will increase the image resolution with respect
        to the figure's layout pixel dimensions. Whereas as scale factor of
        less than 1.0 will decrease the image resolution.

        If not specified, will default to:
            - `plotly.io.kaleido.scope.default_scale` if engine is "kaleido"
            - `plotly.io.orca.config.default_scale` if engine is "orca"

    validate: bool
        True if the figure should be validated before being converted to
        an image, False otherwise.

    engine: str
        Image export engine to use:
         - "kaleido": Use Kaleido for image export
         - "orca": Use Orca for image export
         - "auto" (default): Use Kaleido if installed, otherwise use orca

    Returns
    -------
    None
    """
    assert fig is not None or image_bytes is not None

    # Try to cast `file` as a pathlib object `path`.
    # ----------------------------------------------
    if isinstance(file, string_types):
        # Use the standard Path constructor to make a pathlib object.
        path = Path(file)
    elif isinstance(file, Path):
        # `file` is already a Path object.
        path = file
    else:
        # We could not make a Path object out of file. Either `file` is an open file
        # descriptor with a `write()` method or it's an invalid object.
        path = None

    # Infer format if not specified
    # -----------------------------
    if path is not None and format is None:
        ext = path.suffix
        if ext:
            format = ext.lstrip(".")
        else:
            raise ValueError(
                """
Cannot infer image type from output path '{file}'.
Please add a file extension or specify the type using the format parameter.
For example:

    >>> import plotly.io as pio
    >>> pio.write_image(fig, file_path, format='png')
""".format(
                    file=file
                )
            )

    # Request image
    # -------------
    # Do this first so we don't create a file if image conversion fails
    if image_bytes is None :
        img_data = to_image(
            fig,
            format=format,
            scale=scale,
            width=width,
            height=height,
            validate=validate
        )
    else :
        img_data = image_bytes
    # Open file
    # ---------
    if path is None:
        # We previously failed to make sense of `file` as a pathlib object.
        # Attempt to write to `file` as an open file descriptor.
        try:
            file.write(img_data)
            return
        except AttributeError:
            pass
        raise ValueError(
            """
The 'file' argument '{file}' is not a string, pathlib.Path object, or file descriptor.
""".format(
                file=file
            )
        )
    else:
        # We previously succeeded in interpreting `file` as a pathlib object.
        # Now we can use `write_bytes()`.
        path.write_bytes(img_data)


import cv2

# import io
# from PIL import Image

def img_bytes_to_array(image_bytes) :
    return cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    #return np.array(Image.open(io.BytesIO(image_bytes))) 

def figure_to_images(fig, file=None, format=None, width=None, height=None, scale=None, validate=True):
    image_bytes = to_image(fig=fig, format=format, width=width, height=height, scale=scale, validate=validate)
    if file is not None : write_image(file, image_bytes)
    return img_bytes_to_array(image_bytes)



#################################################

def visualize_embeddings(model, dataloader, max_sample = float("inf"), N = 1000) :
    if False :
        batch = next(iter(dataloader))
        E_a, E_b, tensor = [], [], []
        m = 0
        for batch in dataloader :
            _, e_a, e_b, t, _ = model._get_reconstruction_loss(batch)
            E_a.append(e_a)
            E_b.append(e_b)
            tensor.append(t)
            m = m + e_a.size(0)
            if m >= max_sample : break

        E_a = torch.cat(E_a, dim=0).detach().cpu()
        E_b = torch.cat(E_b, dim=0).detach().cpu()
        tensor = torch.cat(tensor, dim=0).detach().cpu()

        E = E_a + E_b if model.operator == "+" else E_a*E_b
        E = E / model.E_factor
        if E.size(1) > 2 : E = torch.from_numpy(PCA(random_state=0).fit_transform(E)[:,:2])

        #x, y = torch.cartesian_prod(words, words).unbind(dim=1)
        x, y = E.unbind(dim=1)
    else :
        E = model.embeddings.weight.detach().cpu()
        A = E.min(dim=0).values 
        B = E.max(dim=0).values
        x, y = np.linspace(start = A[0], stop = B[0], num=N), np.linspace(start = A[1], stop = B[1], num=N)
        xy = np.array(np.meshgrid(x, y)).T.reshape(-1,2) # (N*N,2)
        #x, y = xy[:,0], xy[:,1]
        #tensor = model.mlp(torch.from_numpy(xy).float()).squeeze()
        xy = torch.from_numpy(xy).float()
        x, y = xy.unbind(dim=1)
        tensor = model.mlp(xy).squeeze()


    if not model.hparams.regression : z = tensor.softmax(dim=1).argmax(dim=1)
    else : z = tensor#.round()

    fig = go.Figure(
        data=go.Contour(
          x=x, y=y, z=z,        
          #contours=dict(coloring='lines',showlabels=True,),
          contours=dict(
              coloring ='heatmap',
              showlabels = True, # show labels on contours
              labelfont = dict( # label font properties
                              size = 12,
                              color = 'white',
                          )
              )
        )
    )
    
    fig.show()

if __name__ == "__main__":
    from .utils import AttrDict
    from .dataset import get_dataloader
    from .modeling import Model

    p =  40
    params = AttrDict({
        "p":p, 
        "emb_dim" : 5, 
        "hidden_dim" : 105,  
        "n_layers" : 2,
        "regression" : False,
        "operator":"+",
        "modular" : True,
        "ID_params" : {},
        "pad_index" : None, 
        "use_wandb" : False
    })

    train_pct = 80
    train_loader, val_loader, dataloader, _ = get_dataloader(p, train_pct, regression = params.regression, operator=params.operator, 
                                                        modular = params.modular, batch_size=256, num_workers=2)
    model = Model(params)

    word_vectors = model.mlp[-1].weight
    _ = display_pca_scatterplot(word_vectors.size(0), model, word_vectors = word_vectors, dim=2, pca=True, title = "layer_layer_weight")

    fig = display_pca_scatterplot(p, model, dim=2, pca=True, title = "Test", show = False)
    #_ = display_pca_scatterplot(p, model.to(device), dim=2, pca=False, perplexity=5, learning_rate=500, iteration=10000)

    image_bytes = to_image(fig, format=None, width=None, height=None, scale=None, validate=True)
    print(image_bytes)

    filename = "../c.png"

    write_image(filename, image_bytes)

    image = figure_to_images(fig, file=filename, format=None, width=None, height=None, scale=None, validate=True)
    print(image.shape)

    #wandb.Image(image)

    word_vectors = model.embeddings.weight
    _ = display_pca_scatterplot(p, model, word_vectors = word_vectors, dim=2, pca=True, title="embeddings")

    word_vectors = model.mlp[-1].weight
    _ = display_pca_scatterplot(word_vectors.size(0), model, word_vectors = word_vectors, dim=2, pca=True, title = "layer_layer_weight")