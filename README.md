# ClassAwareLoss
This code is a PyTorch implementation of ClassAwareLoss proposed in the [Class-aware fish species recognition using deep learning for an imbalanced dataset]( https://www.mdpi.com/1424-8220/22/21/8268) (Published in MDPI Sensors journal, 2022).

Due to sponsorship restrictions, I cannot share the complete code or datasets. The loss was proposed to handle class imbalance issues in a dataset. However, like the focal loss function, you can use the class-aware loss as a general classification loss. As shown in the equation, you can also use the weighting terms for existing localization loss. The loss penalizes samples of the dominant class, gives more weight to the minority class, and updates the weights of each class based on the occurrence of each class instance to reduce the biasedness of the model prediction towards the dominant class. 

![equations](https://github.com/Simeon340703/ClassAwareLoss/assets/50320484/279d8170-4bcb-4087-9d66-0dc118b2ca13),


where Lclsa is class-aware classification loss and Lloca is class-aware localization loss. ns
is the number of training instances per species, n is the total training samples, and η is a
hyper-parameter. We use η = 4 for this training. ns << n.

```bash
If you find this work useful, please cite:
@article{alaba2022class,
  title={Class-aware fish species recognition using deep learning for an imbalanced dataset},
  author={Alaba, Simegnew Yihunie and Nabi, MM and Shah, Chiranjibi and Prior, Jack and Campbell, Matthew D and Wallace, Farron and Ball, John E and Moorhead, Robert},
  journal={Sensors},
  volume={22},
  number={21},
  pages={8268},
  year={2022},
  publisher={MDPI}
}
```

     
