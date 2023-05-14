# ClassAwareLoss
This code is a PyTorch implementation of ClassAwareLoss used in the "Class-aware fish species recognition using deep learning for an imbalanced dataset" paper. https://www.mdpi.com/1424-8220/22/21/8268

Due to sponsorship restrictions, I cannot share code or processed datasets. Like the focal loss function, you can use the class-aware loss as a general classification loss. You can also use the weighting terms for existing localization loss, as shown in the equation.  

![equations](https://github.com/Simeon340703/ClassAwareLoss/assets/50320484/279d8170-4bcb-4087-9d66-0dc118b2ca13)


where Lclsa is class-aware classification loss and Lloca is class-aware localization loss. ns
is the number of training instances per species, n is the total training samples, and η is a
hyper-parameter. We use η = 4 for this training. ns << n.

     
