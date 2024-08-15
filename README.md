# ResNet-From-Scratch
### Implementing Residual Networks from scratch
Deeper network means having better feature extraction. But in CNNs, if we make our network deeper (adding more layers) it will have Degradation issue. <br>
<br>
In Residual Nets, instead of learning the desired underlying mapping *H(x)* , the net learns the residual mapping *F(x) = H(x) — x*. So now the function *H(x) = F(x) + x*. This method of adding the input to the output of the block is also known as shortcut connections. <br> Here is the link to the original ResNet paper: https://arxiv.org/abs/2011.12960

### My article:
**For full explanation of Residual Networks, you can go and checkout my ResNet article on Medium via the following link:**
https://medium.com/@YasinShafiei/residual-networks-resnets-with-implementation-from-scratch-713b7c11f612

### Training the ResNet:
I trained the ResNet-101 model we implemented on the CIFAR-10 dataset (with batch size of 64) for 50 epochs. I used the ADAM optimizer (the original paper uses SGD). I also made the learning rate to be divided by 10 if the average loss of the previous 3 epochs was lower than the current epoch. Here is the loss and accuracy plot of the training:
![training_validation_plots](https://github.com/user-attachments/assets/1df934b4-7bfe-4b93-9cac-bec26b63cdd4)
