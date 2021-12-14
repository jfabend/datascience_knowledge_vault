[torch.optim — PyTorch master documentation](https://pytorch.org/docs/master/optim.html)

[[Optimizers]]

```python
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

optimizer = optim.SGD(net.parameters(), lr=0.001)
```