"""
See UFA_CogWorks.py for the original code with abbreviated comments
See UFA_commented.py for the original code with a start-to-end guide and in-depth explanations

"""






def true_f(x): 
    return np.cos(x)  



class Model:
    
        
    def __init__(self, w, b, v):
        self.w = w
        self.b = b
        self.v = v

    
    def __call__(self, x, activation):
        if activation == "sigmoid":
            sig_out = sigmoid(x @ self.w + self.b) # np.matmul and np.add both return a new tensor (not in-place modification)
            out = sig_out @ self.v  
        elif activation == "relu":
            sig_out = relu(x @ self.w + self.b) 
            out = sig_out @ self.v  
        else:
            print("Error: invalid activation function")
        return out

    
    @property
    def parameters(self):
        return (self.w, self.b, self.v)

    
    def load_parameters(self, w, b, v):

        self.w = w
        self.b = b
        self.v = v



def l1_loss(pred, true):
  
    diff = pred - true  
    abs_diff = np.abs(diff)  
    mean_abs_diff = np.mean(abs_diff) 
    return mean_abs_diff 



def gradient_step(tensors, learning_rate):
    """
    Performs gradient-step in-place on each of the provides tensors 
    according to the standard formulation of gradient descent.

    Parameters
    ----------
    tensors : Union[Tensor, Iterable[Tensors]]
        A single tensor, or an iterable of an arbitrary number of tensors.

        If a `tensor.grad` is `None`for a specific tensor, the update on
        that tensor is skipped.

    learning_rate : float
        The "learning rate" factor for each descent step. A positive number.

    Notes
    -----
    The gradient-steps performed by this function occur in-place on each tensor,
    thus this function does not return anything
    """
    if isinstance(tensors, mg.Tensor):
        tensors = [tensors]

    for t in tensors:
        if t.grad is not None:
            t.data -= learning_rate * t.grad





"""--------PLOTS--------"""





def Plot_Activation_Functions():

    # plotting sigmoid(x)

    fig, ax = plt.subplots()
    x = np.linspace(-10, 10, 1000)  # <COGSTUB> use np.linspace to create 1,000 evenly-spaced points between [-10, 10]$\varepsilon$
    y = sigmoid(x)  # <COGSTUB> evaluate the sigmoid function for all `x` values. 
    ax.plot(x, y)
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("sigmoid(x)")
    plt.show()

    # plotting relu
    fig, ax = plt.subplots()
    x = np.linspace(-10, 10, 1000)
    y = relu(x)
    ax.plot(x, y)
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("relu(x)")
    plt.show()






def Plot_Approximation_Functions(model_sigmoid, model_relu, train_data):

    y_true = true_f(train_data)  # Compute the true values – true_f(x) – for all `x` in `train_data`

    # no long need to do in tandem bc the parameters w v b for sigmoid and relu have been set. 
    # now we're just plotting all points to compare to the loss

    # sigmoid
    fig_sigmoid, ax_sigmoid = plt.subplots()

    y_pred_sigmoid = model_sigmoid(train_data, "sigmoid")  # Compute the predicted values – model(x) – for all `x` in `train_data`


    ax_sigmoid.plot(train_data, y_true, label="True function: f(x)")
    ax_sigmoid.plot(train_data, y_pred_sigmoid, label="Sigmoid approximating function: F(x)")
    ax_sigmoid.grid()
    ax_sigmoid.set_xlabel("x")
    ax_sigmoid.legend()

    plt.show()

    #relu

    fig_relu, ax_relu = plt.subplots()

    y_pred_relu = model_relu(train_data, "relu")

    ax_relu.plot(train_data, y_true, label="True function: f(x)")
    ax_relu.plot(train_data, y_pred_relu, label="ReLU approximating function: F(x)")
    ax_relu.grid()
    ax_relu.set_xlabel("x")
    ax_relu.legend()

    plt.show()








def Plot_Moving_Graph(start_w_sigmoid, start_b_sigmoid, start_v_sigmoid, start_w_relu, start_b_relu, start_v_relu, 
                      params_sigmoid, params_relu):


    #THIS IS HTE MOVING GRPAH TIHNG
    #

    from matplotlib.animation import FuncAnimation

    x = np.linspace(-4 * np.pi, 4 * np.pi, 1000)

    #sigmoid
    fig, ax = plt.subplots()
    ax.plot(x, np.cos(x))
    ax.set_ylim(-2, 2)
    plt.title("Sigmoid")

    _model_sigmoid = Model(start_w_sigmoid, start_b_sigmoid, start_v_sigmoid)
    _model_sigmoid.load_parameters(*params_sigmoid[0])
    (im,) = ax.plot(x.squeeze(), _model_sigmoid(x[:, np.newaxis], "sigmoid").squeeze())


    def update(frame):
        _model_sigmoid.load_parameters(*params_sigmoid[frame])
        im.set_data(x.squeeze(), _model_sigmoid(x[:, np.newaxis], "sigmoid").squeeze())
        return (im,)


    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, len(params_sigmoid)),
        interval=20,
        blit=True,
        repeat=True,
        repeat_delay=1000,
    )

    plt.show()


    #relu

    fig, ax = plt.subplots()
    ax.plot(x, np.cos(x))
    ax.set_ylim(-2, 2)
    plt.title("ReLU")

    _model_relu = Model(start_w_relu, start_b_relu, start_v_relu)
    _model_relu.load_parameters(*params_relu[0])
    (im,) = ax.plot(x.squeeze(), _model_relu(x[:, np.newaxis], "relu").squeeze())


    def update(frame):
        _model_relu.load_parameters(*params_relu[frame])
        im.set_data(x.squeeze(), _model_relu(x[:, np.newaxis], "relu").squeeze())
        return (im,)


    ani = FuncAnimation(
        fig,
        update,
        frames=range(0, len(params_relu)),
        interval=20,
        blit=True,
        repeat=True,
        repeat_delay=1000,
    )

    plt.show()





def Plot_Overlaid_Neurons(num_neurons, model_sigmoid, model_relu):

    #sigmoid
    fig_sigmoid, axes_sigmoid = plt.subplots(ncols=2, nrows=num_neurons // 2)

    x = np.linspace(-2 * np.pi, 2 * np.pi)  # <COGSTUB> create a shape-(50,) array evenly spaces on [-2pi, 2pi]

    # `axes`, `model.w.data`, and `model.b.data` are all 2D numpy arrays, and each
    # store (N,) elements
    #
    # Use the .ravel() method on each of these to transform them into flat arrays/tensors
    # of shape-(N,)

    # For each i in [0, 1, ..., N-1] plot sigmoid(x * w_i + b_i) in the ith `axes` object


    flat_axes = axes_sigmoid.ravel()  # use .ravel() to make shape-(N,)
    flat_w = model_sigmoid.w.ravel()  # use .ravel() to make shape-(N,)
    flat_b = model_sigmoid.b.ravel()  # use .ravel() to make shape-(N,)

    for i in range(num_neurons):
        ax = flat_axes[i]  # get axis-i
        w = flat_w[i]   # get w-i
        b = flat_b[i]  # get b-i
        
        # x is a shape-(50,) array of values evenly-spaced between [-2pi, 2pi]
        # w is a single number
        # b is a single number
        
        sig_out = sigmoid(x * w + b)  # compute the output of a single neuron
        ax.plot(x, sig_out)
        ax.set_ylim(0, 1)

    for ax in axes_sigmoid.ravel():
        ax.grid("True")
        ax.set_ylim(0, 1)
        fig, ax = plt.subplots()

    x = np.linspace(-2 * np.pi, 2 * np.pi)

    # plots the full model output as a thick dashed black curve
    ax.plot(
        x,
        model_sigmoid(x[:, np.newaxis], "sigmoid"),
        color="black",
        ls="--",
        lw=4,
        label="full model output",
    )

    # Add to the plot the scaled activation for each neuron: v σ(x * w + b)
    # Plot each of these using the same instance of `Axes`: `ax`.

    # Use the same code as in the previous cell, but include v_i in your calculation
    # of the neuron's activation

    for i in range(num_neurons):
        w = model_sigmoid.w.ravel()[i]
        b = model_sigmoid.b.ravel()[i]
        v = model_sigmoid.v.ravel()[i]
        
        ax.plot(x,  v * sigmoid(x * w + b))  # plots the activation pattern for that neuron

    ax.grid("True")
    ax.legend()
    ax.set_title("Visualizing the 'activity' of each of the model's scaled neurons")
    ax.set_xlabel(r"$x$")

    plt.show()






    #relu


    fig_relu, axes_relu = plt.subplots(ncols=2, nrows=num_neurons // 2)

    x = np.linspace(-2 * np.pi, 2 * np.pi)
    flat_axes = axes_relu.ravel()  # use .ravel() to make shape-(N,)
    flat_w = model_relu.w.ravel()  # use .ravel() to make shape-(N,)
    flat_b = model_relu.b.ravel()  # use .ravel() to make shape-(N,)

    for i in range(num_neurons):
        ax = flat_axes[i]  # get axis-i
        w = flat_w[i]   # get w-i
        b = flat_b[i]  # get b-i
        
        # x is a shape-(50,) array of values evenly-spaced between [-2pi, 2pi]
        # w is a single number
        # b is a single number
        
        relu_out = relu(x * w + b)  # compute the output of a single neuron
        ax.plot(x, relu_out)
        ax.set_ylim(0, 1)

    for ax in axes_relu.ravel():
        ax.grid("True")
        ax.set_ylim(0, 1)
        fig, ax = plt.subplots()

    x = np.linspace(-2 * np.pi, 2 * np.pi)

    # plots the full model output as a thick dashed black curve
    ax.plot(
        x,
        model_relu(x[:, np.newaxis], "relu"),
        color="black",
        ls="--",
        lw=4,
        label="full model output",
    )


    # Add to the plot the scaled activation for each neuron: v σ(x * w + b)
    # Plot each of these using the same instance of `Axes`: `ax`.

    # Use the same code as in the previous cell, but include v_i in your calculation
    # of the neuron's activation

    for i in range(num_neurons):
        w = model_relu.w.ravel()[i]
        b = model_relu.b.ravel()[i]
        v = model_relu.v.ravel()[i]
        
        ax.plot(x,  v * relu(x * w + b))  # plots the activation pattern for that neuron

    ax.grid("True")
    ax.legend()
    ax.set_title("Visualizing the 'activity' of each of the model's scaled neurons")
    ax.set_xlabel(r"$x$")

    plt.show()







"""--------MAIN WITH PLOTS--------"""


def Main_With_All_Plots():

    from noggin import create_plot
    plotter_sigmoid, fig_sigmoid, ax_sigmoid = create_plot(metrics=["sigmoid_loss"])
    plotter_relu, fig_relu, ax_relu = create_plot(metrics=["relu_loss"])
    ax_sigmoid.set_ylim(0, 1)
    ax_relu.set_ylim(0, 1)

    train_data = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(1000, 1)

    num_neurons = 10
    # need to specify to different sets because otherwise modifying one while doing calculations changes the other
    # pass by reference issues
    start_w_sigmoid = normal(1, num_neurons)  # default dtype=np.float32
    start_b_sigmoid = normal(num_neurons, )
    start_v_sigmoid = normal(num_neurons, 1)

    # convert to 64 to create a copy by value, then back to 32 to retain dtype but now in a new tensor
    start_w_relu = mg.astensor(mg.astensor(start_w_sigmoid, dtype=np.float64), dtype=np.float32)
    start_b_relu = mg.astensor(mg.astensor(start_b_sigmoid, dtype=np.float64), dtype=np.float32)
    start_v_relu = mg.astensor(mg.astensor(start_v_sigmoid, dtype=np.float64), dtype=np.float32)


    model_sigmoid = Model(start_w_sigmoid, start_b_sigmoid, start_v_sigmoid)

    model_relu = Model(start_w_relu, start_b_relu, start_v_relu)  


    params_sigmoid = [] 
    params_relu = []

    batch_size = 25
    lr = 0.01 

    for epoch_cnt in range(1000):  

        idxs = np.arange(len(train_data))

        np.random.shuffle(idxs)


        if epoch_cnt % 10 == 0:
            params_sigmoid.append([w.data.copy() for w in [model_sigmoid.w, model_sigmoid.b, model_sigmoid.v]])
            params_relu.append([w.data.copy() for w in [model_relu.w, model_relu.b, model_relu.v]])

        for batch_cnt in range(0, len(train_data) // batch_size):

            batch_indices = idxs[batch_cnt * batch_size : (batch_cnt + 1) * batch_size]

            batch = train_data[batch_indices] 

            predictions_sigmoid = model_sigmoid(batch, "sigmoid") 
            predictions_relu = model_relu(batch, "relu") 
            # it is okay to pass batch into both without duplicating, because batch is not modified by Model.__call__()

            truth = true_f(batch) 

            loss_sigmoid = l1_loss(predictions_sigmoid, truth)
            loss_relu = l1_loss(predictions_relu, truth)

            loss_sigmoid.backward()
            loss_relu.backward()

            gradient_step(model_sigmoid.parameters, lr)
            gradient_step(model_relu.parameters, lr) 

            plotter_sigmoid.set_train_batch({"sigmoid_loss": loss_sigmoid.item()}, batch_size=batch_size) 
            plotter_relu.set_train_batch({"relu_loss": loss_relu.item()}, batch_size=batch_size) 


        plotter_sigmoid.set_train_epoch() 
        plotter_relu.set_train_epoch()


    plotter_sigmoid.plot()
    plotter_relu.plot()
    plotter_sigmoid.show()
    plotter_relu.show()


    #____________PLOTS____________

    Plot_Activation_Functions()

    #Plot_Overlaid_Neurons(num_neurons, model_sigmoid, model_relu)

    #Plot_Moving_Graph(start_w_sigmoid, start_b_sigmoid, start_v_sigmoid, start_w_relu, start_b_relu, start_v_relu, 
                     # params_sigmoid, params_relu)

    Plot_Approximation_Functions(model_sigmoid, model_relu, train_data)







"""----------MAIN----------"""

import numpy as np
import mygrad as mg
import matplotlib.pyplot as plt
from mygrad.nnet.initializers.normal import normal
from mygrad.nnet import sigmoid, relu

Main_With_All_Plots()