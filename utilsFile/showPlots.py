import transformer_lens.utils as utils
import plotly.express as px

def imshow(tensor,render=False, renderer=None, save='head.html', **kwargs):
    fig = px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", **kwargs)
    if render:
        fig.show(renderer)
    fig.write_html(save)

def line(tensor, renderer=None,render=False, save='accuracy.html', **kwargs):
    fig = px.line(y=utils.to_numpy(tensor), **kwargs)
    if render:
        fig.show(renderer)
    fig.write_html(save)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)