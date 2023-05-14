# ClassAwareLoss
This code is a PyTorch implementation of ClassAwareLoss used in the "Class-aware fish species recognition using deep learning for an imbalanced dataset" paper. https://www.mdpi.com/1424-8220/22/21/8268
\begin{equation}
 \label{class_aware}
 \begin{aligned}
     L_{clsa} &=  \frac{1 - \left ( \frac{n_s}{n} \right )}{1 -\left ( \frac{{n_s}}{n} \right )^\eta }L_{cls},\\
     L_{loca} &= \frac{1 - \left ( \frac{n_s}{n} \right )}{1 -\left ( \frac{{n_s}}{n} \right )^\eta }L_{loc},
     \end{aligned}
 \end{equation}
