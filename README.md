
DataSet : https://cdn.cs50.net/ai/2020/x/projects/5/gtsrb.zip

I tried various combinations of activations functions and optimizers. The "Linear" activation function produced high accuracy and was quicker,
but caused an increased amount of loss. For the output layer, most of the other activation functins decreased accuracy drastically.
All the other optimizers except "Adam" returned inefficient results, so I stuck with "relu" and "softmax" as activation functions.
After this I began to read what each of these functions mean and understood why they returned such results.
To find the balance between accuracy, loss and time I convoluted the images after pooling them again.

I used 32 filters on a 4x4 kernel for the first convolution, then I pooled the images with a 2x2 pool size.
I convoluted the images again with 64 filters on a 4x4 kernel and again pooled them with a 2x2 pool size.
After flattening, I added 144 hidden layers and left the dropout at 0.5. Reducing the dropout didn't change the results a lot so I left it at 0.5.
I used 43 output layers as specified an used softmax as the activation function for it.
Accuracy is over 97% and loss is less 1% and runtime is 1 second for 333 samples. The step time per sample increased during the last run, 
from 12ms to 15ms but then figured it was due to the screen recorder using the resources.