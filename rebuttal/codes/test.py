import torch
import torch.nn as nn
import torch.optim as optim

# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.fc = nn.Linear(1, 1)
#
#     def forward(self, x):
#         return self.fc(x)
#
# model = SimpleNet()
# loss_function = nn.MSELoss()
# # opt = optim.Adam(model.parameters(), lr=0.01)
#
# # Example input and target
# x = input = torch.randn(1, 1)
# print("x:", input)
# y = output = model(input)
# print("y:", output)
#
# target = torch.zeros_like(input)
# target_2 = torch.ones_like(input)
#
# loss_pre = loss_function(model(input), target_2)
# model.zero_grad()
# loss_pre.backward(retain_graph=True)
#
# loss = loss_function(output, target)
# # First gradient calculation
# model.zero_grad()
# loss.backward(create_graph=True)
#
# # Example: L2 norm of the gradients of the first layer
# grads = model.fc.weight.grad
# print("g': ", grads, 2 * x * y)
# old_grads = grads.detach().clone()
# # grad_function = torch.sum(grads.clone() ** 2)
# grad_list = [grads]
# grad_list = torch.cat(grad_list).view(-1)
# grad_function = torch.sum(grad_list.clone() ** 2)
#
# for item in model.parameters():
#     item.grad.data.zero_()
# # model.fc.weight.grad.data.zero_()
# # opt.zero_grad()
# # model.zero_grad()
#
# # Calculate second-order gradient (gradient of the grad_function)
# grad_function.backward()
# new_grads = model.fc.weight.grad
# print("g'': ", new_grads, x, y, old_grads, 8 * x * x * x * y + old_grads, 8 * x * x * x * y)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
loss_function = nn.MSELoss()

# Example input and target
x = input = torch.randn(1, 2)
print("x:", input)
y = output = model(input)
print("y:", output)

target = torch.zeros_like(output)
target_2 = torch.ones_like(output)

loss_pre = loss_function(model(input), target_2)
model.zero_grad()
loss_pre.backward(retain_graph=True)

loss = loss_function(output, target)
# First gradient calculation
model.zero_grad()
loss.backward(create_graph=True)

# Example: L2 norm of the gradients of the first layer
grads = model.fc.weight.grad
print("g': ", grads, 2 * x * y)
old_grads = grads.detach().clone()
# grad_function = torch.sum(grads.clone() ** 2)
grad_list = [grads]
grad_list = torch.cat(grad_list).view(-1)
grad_function = (grad_list * torch.tensor([0.5, 0.5])).sum()

for item in model.parameters():
    item.grad.data.zero_()
# model.fc.weight.grad.data.zero_()
# opt.zero_grad()
# model.zero_grad()

# Calculate second-order gradient (gradient of the grad_function)
grad_function.backward()
new_grads = model.fc.weight.grad
print("g'': ", new_grads, x, y, x[0][0] * (x[0][0] + x[0][1]), x[0][1] * (x[0][0] + x[0][1]))