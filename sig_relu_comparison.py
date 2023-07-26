
import numpy as np
import mygrad as mg
import matplotlib.pyplot as plt
from mygrad.nnet.initializers.normal import normal
from mygrad.nnet import sigmoid, relu
from noggin import create_plot

def One_Sample():

    plotter_sigmoid, fig_sigmoid, ax_sigmoid = create_plot(metrics=["sigmoid_loss"])
    plotter_relu, fig_relu, ax_relu = create_plot(metrics=["relu_loss"])
    ax_sigmoid.set_ylim(0, 1)
    ax_relu.set_ylim(0, 1)


    train_data = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(1000, 1)

    num_neurons = 10

    start_w_sigmoid = normal(1, num_neurons)
    start_b_sigmoid = normal(num_neurons, )
    start_v_sigmoid = normal(num_neurons, 1)
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