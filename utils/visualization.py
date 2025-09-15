from typing import List, Optional
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np

def plot_train_val_curve(train_loss_per_epoch: List, val_loss_per_epoch: List):
        x = np.arange(len(train_loss_per_epoch))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=train_loss_per_epoch, mode='lines', name='Train Loss'))
        fig.add_trace(go.Scatter(x=x, y=val_loss_per_epoch, mode='lines', name='Validation Loss'))
        
        fig.update_layout(
                title="Train vs Validation Loss",
                xaxis_title="Epochs",
                yaxis_title="Loss"
        )
        fig.show()


def plot_activations_histogram(network_layers: List, 
                             layer_names: Optional[List[str]] = None,
                             bins: int = 50):
    """
    Plot activation histograms for each layer.
    """
    if layer_names is None:
        layer_names = [f'Layer {i+1}' for i in range(len(network_layers))]
    
    indices_to_plot = sorted(list(set([0, len(network_layers) // 2, len(network_layers) - 1])))
    layers_to_plot = [network_layers[i] for i in indices_to_plot]
    layer_names_to_plot = [layer_names[i] for i in indices_to_plot]

    n_layers = len(layers_to_plot)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig = sp.make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=layer_names_to_plot
    )
    
    for i, layer in enumerate(layers_to_plot):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1
        
        if hasattr(layer, 'a') and layer.a is not None:
            activations = layer.a.flatten()
            
            fig.add_trace(
                go.Histogram(
                    x=activations,
                    nbinsx=bins,
                    showlegend=False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title='Activation Histograms',
        template='plotly_white'
    )
    
    fig.show()

def plot_gradients_histogram(network_layers: List,
                           layer_names: Optional[List[str]] = None,
                           bins: int = 50,
                           plot_weights: bool = True,
                           plot_biases: bool = True):
    """
    Plot gradient histograms for weights and/or biases.
    """
    if layer_names is None:
        layer_names = [f'Layer {i+1}' for i in range(len(network_layers))]

    indices_to_plot = sorted(list(set([0, len(network_layers) // 2, len(network_layers) - 1])))
    layers_to_plot = [network_layers[i] for i in indices_to_plot]
    layer_names_to_plot = [layer_names[i] for i in indices_to_plot]
    
    plots_per_layer = int(plot_weights) + int(plot_biases)
    total_plots = len(layers_to_plot) * plots_per_layer
    
    if total_plots == 0:
        return
    
    n_cols = min(3, total_plots)
    n_rows = (total_plots + n_cols - 1) // n_cols
    
    subplot_titles = []
    for layer_name in layer_names_to_plot:
        if plot_weights:
            subplot_titles.append(f'{layer_name} - Weights')
        if plot_biases:
            subplot_titles.append(f'{layer_name} - Biases')
    
    fig = sp.make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=subplot_titles
    )
    
    plot_idx = 0
    
    for layer in layers_to_plot:
        # Plot weight gradients
        if plot_weights:
            row = (plot_idx // n_cols) + 1
            col = (plot_idx % n_cols) + 1
            
            if hasattr(layer, 'theta_grad') and layer.theta_grad is not None:
                gradients = layer.theta_grad.flatten()
                
                fig.add_trace(
                    go.Histogram(
                        x=gradients,
                        nbinsx=bins,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            plot_idx += 1
        
        # Plot bias gradients
        if plot_biases:
            row = (plot_idx // n_cols) + 1
            col = (plot_idx % n_cols) + 1
            
            if hasattr(layer, 'bias_grad') and layer.bias_grad is not None:
                gradients = layer.bias_grad.flatten()
                
                fig.add_trace(
                    go.Histogram(
                        x=gradients,
                        nbinsx=bins,
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            plot_idx += 1
    
    fig.update_layout(
        title='Gradient Histograms',
        template='plotly_white'
    )
    
    fig.show()

def plot_weights_histogram(network_layers: List,
                             layer_names: Optional[List[str]] = None,
                             bins: int = 50):
    """
    Plot weights histograms for each layer.
    """
    if layer_names is None:
        layer_names = [f'Layer {i+1}' for i in range(len(network_layers))]

    indices_to_plot = sorted(list(set([0, len(network_layers) // 2, len(network_layers) - 1])))
    layers_to_plot = [network_layers[i] for i in indices_to_plot]
    layer_names_to_plot = [layer_names[i] for i in indices_to_plot]

    n_layers = len(layers_to_plot)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig = sp.make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=layer_names_to_plot
    )

    for i, layer in enumerate(layers_to_plot):
        row = (i // n_cols) + 1
        col = (i % n_cols) + 1

        if hasattr(layer, 'theta'):
            weights = layer.theta.flatten()

            fig.add_trace(
                go.Histogram(
                    x=weights,
                    nbinsx=bins,
                    showlegend=False
                ),
                row=row, col=col
            )

    fig.update_layout(
        title='Weights Histograms',
        template='plotly_white'
    )

    fig.show()