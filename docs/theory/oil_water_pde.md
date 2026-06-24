# Oil-Water Vapor Barrier PDEs

## Plasma ("oil")

$$\frac{\partial n_p}{\partial t} = \nabla \cdot (D_p \nabla n_p) - \alpha n_p n_v$$

## Vapor ("water")

$$\frac{\partial n_v}{\partial t} = \nabla \cdot (D_v \nabla n_v) - v_v \nabla n_v + \alpha n_p n_v - S_{breed}$$

## Tritium breeding

$$\nabla \cdot (-D_T \nabla n_T + v_v n_T) = \sigma_{Li} \Phi_n n_v$$

## Interface thickness

$$\delta = \left(\frac{D_p D_v}{\alpha^2 n_0 n_{wall}}\right)^{1/4}$$

## Extraction criterion

$$v_v > \frac{D_T}{\delta} \quad \Rightarrow \quad Pe > 1$$
