Paper	Scaling Law	  # points
Chinchilla scaling: A replication attempt (chinchilla)	$L(N,D) = E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}$	245
Distillation Scaling Laws (distillation)	$L_S(L_T,N_S,D_S) = L_T + \frac{1}{L_T^{c_0}} \left(1 + \left(\frac{L_T}{\left[1.220 + \left(\frac{3355}{N_S^{0.408}} + \frac{18186}{D_S^{0.431}}\right)^{0.452}\right] \cdot d_1}\right)^{1/f_1}\right)^{-c_1 f_1} \left(\frac{A'}{N_S^{\alpha'}} + \frac{B'}{D_S^{\beta'}}\right)^{\gamma'}$	354
Scaling Laws for Upcycling Mixture-of-Experts Language Models (moe_sparsity)	  1. $L(P, N_{\text{active}}) = \left( e^{d_1} \cdot P^{-a} \cdot N_{\text{active}}^{-b} \cdot e^{c \log P \cdot \log N_{\text{active}}} + e^{d_3} \right)$
  2. $L(P, N_{\text{active}}) = e^{d_1} \cdot P^{-a} \cdot N_{\text{active}}^{-b} + e^{d_3}$
  3. $L(P, N_{\text{active}}) = e^{d_1} \cdot P^{-a} + e^{d_2} \cdot N_{\text{active}}^{-b} \cdot e^{c \log P \cdot \log N_{\text{active}}} + e^{d_3}$
  4. $L(P, N_{\text{active}}) = e^{d_1} \cdot P^{-a} + e^{d_2} \cdot N_{\text{active}}^{-b} + e^{d_3}$	480
Scaling and evaluating sparse autoencoders (sae)	$L(n,k) = e^{\alpha + \beta_k \log k + \beta_n \log n + \gamma \log k \log n} + e^{\zeta + \eta \log k}$	45


cost: 
sae: $n^{1.6}$
moe_sparsity: $6 N_{\text{dense}}D_1 + 6N_{\text{active}}D_2$
chinchilla: $6ND$
distillation: $6N_sD_s$

split:
distillation: "NS":1820000000,"DS":38900000000 和 "NS":7750000000,"DS":100500000000 是 test set
sae: n=131072, 262144, 524288 是 test set
chinchilla: 按照 n * d 排序，后 20% 作为 test set