
import deepxde as dde
import numpy as np

def heat_exact(x,t):
  return np.exp((-n**2 * (np.pi)**2 *a * t)/L**2)*np.sin((n*x*np.pi)/L)

def get_examples():
  (x_in,x_fin)= (0.0,L)
  (t_in,t_fin)= (0.0,1.0)

  (x_dim,t_dim)=(256,201)

  x = np.linspace(x_in,x_fin, num= x_dim).reshape(x_dim,1)
  t = np.linspace(t_in,t_fin,num = t_dim).reshape(t_dim,1)
  sol = np.zeros((x_dim,t_dim))
  for i in range(x_dim):
    for j in range(t_dim):
      sol[i][j]=heat_exact(x[i,0],t[j,0])

  np.savez("heat_dataset.npz",x=x,t=t,sol= sol)
#sol is stored as the last one and the column name is given and the correpiojding values are given

#here the orderr must be x,y
def pde(x,y):
  dy_t= dde.grad.jacobian(y,x,i=0,j=1)
  dy_xx= dde.grad.hessian(y,x,i=0,j=0)
  return dy_t- (a*dy_xx)

a = dde.Variable(0.01)
L= 1.0
n=1
get_examples()

def get_test_data():
  data = np.load("heat_dataset.npz")

#
  x,t,label = data['x'],data['t'],data['sol'].T

  xx,tt = np.meshgrid(x,t)
#this taes only one arg
  X = np.vstack((xx.ravel(),tt.ravel())).T # we rae taking thje transpose to make it as column vector
  y = label.ravel()[:,None]
  return X,y,xx,tt

# we need to create the dataset to and and pass t to the ic and bc ti tell tat we are applyion the bc and ic on this
space = dde.geometry.Interval(0,L)
time_int = dde.geometry.TimeDomain(0,1)
domain = dde.geometry.GeometryXTime(space,time_int)

x_obs,y_obs,_,_ = get_test_data()
obs_data = dde.icbc.PointSetBC(x_obs,y_obs,component = 0)
#
bc = dde.icbc.DirichletBC(domain,lambda x: 0 ,lambda _,on_boundary:on_boundary)
#
ic = dde.icbc.IC(domain,lambda x: np.sin((n*np.pi*x[:,0:1])/L),lambda _, on_initial:on_initial)

data = dde.data.TimePDE(domain,pde,[bc,ic,obs_data],num_boundary=60,num_domain=5000,num_initial=150,num_test=5000)
network = dde.nn.FNN([2]+ [30]*3 + [20]*3 + [1],'sin','Glorot normal')
model = dde.Model(data,network)

# giving more weight to the tpde loss so that it could learn physics well
#we need to give l2 to track the l2 orm
model.compile('L-BFGS',lr= 1e-3,external_trainable_variables=[a],loss_weights=[10,1,1,5])
variable_call = dde.callbacks.VariableValue([a], period = 100)
loss_hist , train_state= model.train(iterations = 3000,callbacks=[variable_call]) # calling back vraible

#using L bfgs for refinement
# model.compile('L-BFGS',lr= 1e-3,external_trainable_variables=[a],loss_weights=[1,1,1])
# variable_call = dde.callbacks.VariableValue([a], period = 100)
# loss_hist , train_state= model.train(iterations = 2000,callbacks=[variable_call])

#to print the values in te note
print('Variabel call back values:', variable_call.value)
variable_hist = variable_call.value


print(f'The final alpha value: {a.numpy()}')
dde.saveplot(loss_hist,train_state,isplot = True,issave=True)

X,y,x,t = get_test_data()
x = x.flatten().reshape(-1,1)
t= t.flatten().reshape(-1,1)
y_pred = model.predict(X)


mse = np.mean((y-y_pred)**2)
import matplotlib.pyplot as plt
l2 = np.linalg.norm(y-y_pred)/np.linalg.norm(y)

error = y - y_pred

#contour was pkotted ti anaklyse and find which arer had more error
#plt.contourf(x.reshape(256,201),t.reshape(256,201),error.reshape(256,201),cmap='inferno')
print(f'L2 norm :{l2}')
plt.plot(y,y_pred,'r-')
print(f'mse: {mse}')

model.save('heat_model_weights')

#a_hist = [v[0] for v in variable_hist]
#plt.plot(variable_hist)


# 1. Ensure we have the list of values
# If variable_hist is a list of lists, use [v[0] for v in variable_hist]
# If it's already a flat list, use it directly
data = [v[0] if isinstance(v, (list, np.ndarray)) else v for v in variable_hist]

# 2. Create the iteration steps (period=100)
iterations = range(0, len(data) * 100, 100)

# 3. Plot explicitly
plt.figure(figsize=(8, 4))
plt.plot(iterations, data, marker='o', linestyle='-', color='b')
plt.xlabel("Iterations")
plt.ylabel("Value of Alpha")
plt.title("Convergence of Alpha over Training")
plt.grid(True)
plt.show()

