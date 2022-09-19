from IPython.display import display
import ipywidgets as widgets

features_W = widgets.IntSlider(
    value=6,
    min=4,
    max=12,
    step=1,
    description='features',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
)
display(features_W)
evaluations_W = widgets.IntSlider(
    value=50,
    min=50,
    max=1200,
    step=50,
    description='evaluations',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
)
display(evaluations_W)
xai_model_W = widgets.Select(
    options=['LIME', 'SHAP'],
    value='LIME',
    # rows=10,
    description='Xai model:',
    disabled=False
)
display(xai_model_W)

dataset_W = widgets.Select(
    options=['CIFAR10', 'CIFAR100', 'EMNIST','FashionMNIST'],
    value='CIFAR10',
    # rows=10,
    description='Dataset:',
    disabled=False
)
display(dataset_W)

perturbation_W = widgets.Select(
    options=['square', 'quickshift'],
    value='square',
    description='Perturbation:',
    disabled=False
)
display(perturbation_W)

sigma_W = widgets.FloatSlider(
    value=4,
    min=1,
    max=12,
    step=0.5,
    description='sigma',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
)
display(sigma_W)
maxDist_W = widgets.FloatSlider(
    value=150,
    min=30,
    max=300,
    step=0.5,
    description='max_dist',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
display(maxDist_W)
kernel_W = widgets.IntSlider(
    value=20,
    min=1,
    max=70,
    step=1,
    description='kernel',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='d',
)
display(kernel_W)
ratio_W = widgets.FloatSlider(
    value=0.5,
    min=0.1,
    max=3.0,
    step=0.1,
    description='ratio',
    disabled=False,
    continuous_update=False,
    orientation='horizontal',
    readout=True,
    readout_format='.1f',
)
display(ratio_W)