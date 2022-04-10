import numpy as np
import time
class MLP(object):
    def __init__(self,num_input,num_output):
        self.num_input=num_input
        self.num_output=num_output
        std=0.01
        self.weight = np.random.normal(loc=0.0, scale=std, size=(self.num_input, self.num_output))
        self.bias = np.zeros([1, self.num_output])
    #前向传播计算
    def forward(self,input):
        start_time=time.time()
        self.input=input
        self.output=np.matmul(input,self.weight)+self.bias
        return self.output
    #反向传播计算
    def backward(self,top_diff):
        self.d_weight=np.dot(self.input.T,top_diff)
        self.d_bias=np.sum(top_diff,axis=0)
        bottom_diff=np.dot(top_diff,self.weight.T)
        return bottom_diff
    def update_param(self,lr):
        self.weight=self.weight-lr*self.d_weight
        self.bias=self.bias-lr*self.d_bias
    def load_param(self,weight,bias):
        # assert self.weight.shape==weight.shape
        # assert self.bias.shape==bias.shape
        self.weight=weight
        self.bias=bias
    #参数保存
    def save_param(self):
        return self.weight,self.bias



class Relu(object):
    def __init__(self):
        pass

    def forward(self, input):
        self.input = input
        output = np.maximum(0, input)
        return output

    def backward(self, top_diff):
        bottom_diff = top_diff
        bottom_diff[self.input < 0] = 0
        return bottom_diff

class Softmax(object):
    def __init__(self):
        pass
    def forward(self,input):
        input_max=np.max(input,axis=1,keepdims=True)
        input_exp=np.exp(input-input_max)
        self.prob=input_exp/np.sum(input_exp,axis=1,keepdims=True)
        return self.prob
    def get_loss(self,label):
        self.batch_size=self.prob.shape[0]
        self.label_onehot=np.zeros_like(self.prob)
        self.label_onehot[np.arange(self.batch_size),label]=1.0
        loss = -np.sum(np.log(self.prob)*self.label_onehot/self.batch_size)
        return loss
    def backward(self):
        bottom_diff=(self.prob-self.label_onehot)/self.batch_size
        return bottom_diff

class mymodel(object):
    def __init__(self, num_input, num_hidden, num_output, lr, l2, milestone=500, gamma=0.5):
        self.input_size = num_input
        self.hidden = num_hidden
        self.output_size = num_output
        self.milestone=milestone
        self.gamma=gamma
        self.lr=lr
        self.lr_=lr
        self.l2=l2
        print('building two-layer perception model...')
        self.fc1 = MLP(self.input_size, self.hidden)
        self.relu1 = Relu()
        self.fc2 = MLP(self.hidden, self.output_size)
        self.softmax = Softmax()
        self.update_layer = [self.fc1, self.fc2]

    def forward(self,input):
        h1 = self.fc1.forward(input)
        h1 = self.relu1.forward(h1)
        h2 = self.fc2.forward(h1)
        prob = self.softmax.forward(h2)
        return prob
    def __call__(self,input):
        return self.forward(input)
    def init_model(self):
        print('Initializing parameters of each layer in MLP...')
        for layer in self.update_layer:
            layer.init_param()

    def save_model(self, param_dir):
        print('Saving parameters to file ' + param_dir)
        params = {}
        params['w1'], params['b1'] = self.fc1.save_param()
        params['w2'], params['b2'] = self.fc2.save_param()
        np.save(param_dir, params)

    def load_model(self, param_dir):
        print('Loading parameters from file ' + param_dir)
        params = np.load(param_dir,allow_pickle=True).item()
        self.fc1.load_param(params['w1'], params['b1'])
        self.fc2.load_param(params['w2'], params['b2'])

    def backward(self):  # 神经网络的反向传播
        dloss = self.softmax.backward()
        dh2 = self.fc2.backward(dloss)
        dh1 = self.relu1.backward(dh2)
        dh1 = self.fc1.backward(dh1)

    def update(self, lr):
        for layer in self.update_layer:
            layer.update_param(lr)

    def lr_decay(self, epoch):
        n = int(epoch / self.milestone)
        self.lr_ = self.lr * (self.gamma ** n)


