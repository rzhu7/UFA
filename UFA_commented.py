import numpy as np
import mygrad as mg
import matplotlib.pyplot as plt
from mygrad.nnet.initializers.normal import normal
from mygrad.nnet import sigmoid



"""

SETTING UP LIVE PLOTTING WITH MATPLOTLIB / CHANGING OPTIONAL SETTINGS

https://matplotlib.org/3.2.2/users/shell.html
If you are using IPython, please see "IPython to the rescue"
Otherwise, see "Other python interpreters" or follow along with my steps below

In any terminal, run: matplotlib.matplotlib_fname()
Navigate to the specified path destination
Copy the file "matplotlibrc" into the same folder as this python file
Use any text/code editor to change the following two settings:
"backend" property from [backend: Agg] to [backend: TkAgg]
"interactive" property from [interactive: False] to [interactive: True]

Note that this only allows for live plotting in the current working directory
To see other ways of changing myplotlib settings, see "The matplotlibrc file" in the official documentation below:
https://matplotlib.org/stable/tutorials/introductory/customizing.html#customizing-with-matplotlibrc-files

"""


"""
The below error will pop up while running this code, but it shouldn't impact our graphs (it's simply a default notification)

UserWarning: Live plotting is not supported when matplotlib uses the 'QtAgg' backend. Instead, use the 'nbAgg' backend.
"""



class Model:
    
    # randomize w, b, v to transform 
    def initialize_params(self, num_neurons: int):
  
        self.w = normal(1, num_neurons)  
        self.b = normal(num_neurons)
        self.v = normal(num_neurons, 1)
        

        
    def __init__(self, num_neurons: int):

        self.N = num_neurons  

        self.initialize_params(num_neurons)
    
    def __call__(self, x):

        sig_out = sigmoid(x @ self.w + self.b) 
        out = sig_out @ self.v  
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








""" Main Code """





from noggin import create_plot
plotter, fig, ax = create_plot(metrics=["loss"])
ax.set_ylim(0, 1)



train_data = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(1000, 1)


model = Model(num_neurons=10)  
batch_size = 25

def true_f(x): 
    return np.cos(x)  


params = [] 

















lr = 0.01 


for epoch_cnt in range(1000):  

    idxs = np.arange(len(train_data))

    np.random.shuffle(idxs)


    if epoch_cnt % 10 == 0:
        params.append([w.data.copy() for w in [model.w, model.b, model.v]])

    for batch_cnt in range(0, len(train_data) // batch_size):

        batch_indices = idxs[batch_cnt * batch_size : (batch_cnt + 1) * batch_size]



        """
        
        """

        # note that we choose our data from a train_data set that we expect to have true values for
        batch = train_data[batch_indices] 



        """
        train_data is literally just all the possible x-values (or like 10000 x-values) in the interval we care abt
        aka -2pi , 2pi

        those are the data points at which we will calculate a prediction based on our model
        compare that to the true value

        we have an infinite (10000 but we coulda picked more) number of data points here and we can evaluate an 
        infinite number of true f(x) values based on those because we know f(x)

        in real life, train_data isn't infinite, because we don't know the true function f(x). that's what we're trying to find
        in real life, we just have some discrete collected data points from an experiement or an observational study

        then, we select a random batch of points from that set of data
        we have to make predictions based on those points
        otherwise we will have no true data value to which we can compare our prediction function's output


        """

        # model(batch) is equivalent to model.__call__(batch)

        predictions = model(batch) 


        """DABAO NOTE"""
# im guessing that this model() is really just calling model. __call__
# need to learn how python objects and instance methods work
# i guess this bc batch is a shape(M, 1) tensor just like how __call__ takes an "array-like" object with shape (M, 1)


        # evaluate real f(x) values at the set of data points AKA x-values at which our prediction model made approximations
        # note that we call true_f because we are trying to approximate a known equation, using which we can evaluate f(x) at any point
        # if this were a method to find a line of best fit for a set of discrete data points 
        # (i.e. from a lab experiment, observational study, etc.)
        # we wouldn't have a true_f and would instead compare the approximations to an array/tensor of measured "true" values
        truth = true_f(batch) 

        # Compute the loss associated with our predictions, compared to the true values
        # we're passing two shape-(M, 1) tensors into a function of l1_loss which, using np.mean(), typically accepts shape-(M, )
        # however, l1_loss takes the difference of the two tensors, and addition/subtraction can be done on any two tensors of same shape
        # the final output is of shape-() because we're finding np.mean(), so whether we input (M, ) or (M, 1) doesn't matter
        loss = l1_loss(predictions, truth)


        # differentiate the L-1 function with respect to w, v, b
        loss.backward()


        # depending on the direction of the derivative, adjust said model parameters
        # i.e. if L-1 is decreasing at the current w, increase w to obtain a lower L-1 with the next iteration
        gradient_step-(model.parameters, lr) 


        # ultimately we want to achieve a derivative of 0 (ideally an absolute minimum of the error function)
        #
        # two problems might occur:
        # 1. the function has no derivative of 0 (no convergence)
        # 2. we find a relative minimum as opposed to absolute
        #
        # both cases are the topic of ongoing research with no definitive solution
        # one workaround is to specify a maximum error that we would be satisfied with and stop iterating when the error is less than that
        # otherwise, if L-1 is decreasing so slowly that efficiency is compromised, just stop when |derivative| < min_desired_decrease
        # as worst-case scenario, neural network training models specify a number of epochs after which training ends
        #
        # if this interests you, searching up "plateau phenomenon" is a good place to start
        # this study is quite approachable and interesting, with good visuals: https://www.osti.gov/servlets/purl/1786969




        # note when doing the experiment and running code, make sure to print(error, exited) if we terminate
        #traiing for any of the above reasons
        # ideally this wont happen, and then we can say in the paper that oh there werent confoudngin variables like these




        """DABAO NOTE"""



        plotter.set_train_batch({"loss": loss.item()}, batch_size=batch_size) 
    plotter.set_train_epoch() 


plotter.plot()












































plotter.show()


fig, ax = plt.subplots()

# <COGINST>
y_true = true_f(train_data)  # Compute the true values – true_f(x) – for all `x` in `train_data`
y_pred = model(train_data)  # Compute the predicted values – model(x) – for all `x` in `train_data`

ax.plot(train_data, y_true, label="True function: f(x)")
ax.plot(train_data, y_pred, label="Approximating function: F(x)")
ax.grid()
ax.set_xlabel("x")
ax.legend()
plt.show()












# plotting sigmoid(x)

fig, ax = plt.subplots()

x = np.linspace(-10, 10, 1000)  # <COGSTUB> use np.linspace to create 1,000 evenly-spaced points between [-10, 10]$\varepsilon$

y = sigmoid(x)  # <COGSTUB> evaluate the sigmoid function for all `x` values. 

ax.plot(x, y)

ax.grid()
ax.set_xlabel("x")
ax.set_ylabel("sigmoid(x)")
plt.show()






THIS IS HTE MOVING GRPAH TIHNG
from matplotlib.animation import FuncAnimation

x = np.linspace(-4 * np.pi, 4 * np.pi, 1000)



fig, ax = plt.subplots()
ax.plot(x, np.cos(x))
ax.set_ylim(-2, 2)
_model = Model(model.N)
_model.load_parameters(*params[0])
(im,) = ax.plot(x.squeeze(), _model(x[:, np.newaxis]).squeeze())


def update(frame):
    # ax.figure.canvas.draw()
    _model.load_parameters(*params[frame])
    im.set_data(x.squeeze(), _model(x[:, np.newaxis]).squeeze())
    return (im,)


ani = FuncAnimation(
    fig,
    update,
    frames=range(0, len(params)),
    interval=20,
    blit=True,
    repeat=True,
    repeat_delay=1000,
)

plt.show()




















fig, axes = plt.subplots(ncols=2, nrows=model.N // 2)

x = np.linspace(-2 * np.pi, 2 * np.pi)  # <COGSTUB> create a shape-(50,) array evenly spaces on [-2pi, 2pi]

# `axes`, `model.w.data`, and `model.b.data` are all 2D numpy arrays, and each
# store (N,) elements
#
# Use the .ravel() method on each of these to transform them into flat arrays/tensors
# of shape-(N,)

# For each i in [0, 1, ..., N-1] plot sigmoid(x * w_i + b_i) in the ith `axes` object


flat_axes = axes.ravel()  # use .ravel() to make shape-(N,)
flat_w = model.w.ravel()  # use .ravel() to make shape-(N,)
flat_b = model.b.ravel()  # use .ravel() to make shape-(N,)

for i in range(model.N):
    ax = flat_axes[i]  # get axis-i
    w = flat_w[i]   # get w-i
    b = flat_b[i]  # get b-i
    
    # x is a shape-(50,) array of values evenly-spaced between [-2pi, 2pi]
    # w is a single number
    # b is a single number
    
    sig_out = sigmoid(x * w + b)  # compute the output of a single neuron
    ax.plot(x, sig_out)
    ax.set_ylim(0, 1)

for ax in axes.ravel():
    ax.grid("True")
    ax.set_ylim(0, 1)









    fig, ax = plt.subplots()
x = np.linspace(-2 * np.pi, 2 * np.pi)

# plots the full model output as a thick dashed black curve
ax.plot(
    x,
    model(x[:, np.newaxis]),
    color="black",
    ls="--",
    lw=4,
    label="full model output",
)



# Add to the plot the scaled activation for each neuron: v σ(x * w + b)
# Plot each of these using the same instance of `Axes`: `ax`.

# Use the same code as in the previous cell, but include v_i in your calculation
# of the neuron's activation

for i in range(model.N):
    w = model.w.ravel()[i]
    b = model.b.ravel()[i]
    v = model.v.ravel()[i]
    
    ax.plot(x,  v * sigmoid(x * w + b))  # plots the activation pattern for that neuron

ax.grid("True")
ax.legend()
ax.set_title("Visualizing the 'activity' of each of the model's scaled neurons")
ax.set_xlabel(r"$x$")

