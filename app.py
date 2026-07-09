import deepxde as dde
import numpy as np
import streamlit as st

st.title("Heat Inverse PINN Solver")
x = st.slider("Select x", 0.0, 1.0, 0.5)
t = st.slider("Select t", 0.0, 1.0, 0.5)

@st.cache_resource
def get_model():
    a = dde.Variable(0.01)
    space = dde.geometry.Interval(0,1.0)
    time_int = dde.geometry.TimeDomain(0,1.0)
    geotime = dde.geometry.GeometryXTime(space,time_int)

    network = dde.nn.FNN([2]+ [30]*3 +[20]*3 + [1],'sin','Glorot normal')

    data = dde.data.TimePDE(geotime,None,[],num_domain=0)
    model = dde.Model(data,network)
    input_dummy = np.array([[0.0,0.0]])
    model.compile("adam", lr=0.001,trainable_variable = [a])
    model.predict(input_dummy)
    model.restore("heat_model_weights-717.weights.h5")
    return model,a

model,a = get_model()

alpha = a.numpy()

st.subheader("Model Parameters")
st.metric(label="Identified Thermal Diffusivity (α)", value=f"{alpha_value:.6f}")

def pred_tem(x,t):
# note input kus be 2d array
    input = np.array([[x,t]])
    predicted = model.predict(input)

    return predicted[0][0]

if st.button("Predict Temperature"):
    temp = pred_tem(x, t)
    st.write(f"The predicted temperature is: {temp:.6f}")
