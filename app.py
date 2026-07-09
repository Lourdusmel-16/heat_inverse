import deepxde as dde
import numpy as np

def get_model():

    space = dde.geometry.Tnterval(0,1.0)
    time_int = dde.geometry.TimeInterval(0,1.0)
    geotime = dde.geometry.GeometryXTime(space,time_int)

    network = dde.nn.FNN([2]+ [30]*3 +[20]*3 + [1],'sin','Glorot normal')

    data = dde.data.TimePDE(geotime,None,[],num_domain=0)

    return dde.Model(data,network)

model = get_model()

model.restore("heat_model_weights-3000.ckpt")

def pred_tem(x,t):
# note input kus be 2d array
    input = np.array[[x,t]]
    predicted = model.predict(input)

    return predicted[0][0]

pred = pred_tem(0.5,0.5)
