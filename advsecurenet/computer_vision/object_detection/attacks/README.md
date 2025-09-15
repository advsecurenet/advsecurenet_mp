# Adversarial Attacks

AdvSecureNet supports various adversarial object detection attacks, including:

- DPatch (targeted and untargeted),
- TOG (vanishing, mislabeling (most-likely or least-likely mode), fabrication, untargeted).

Some of these attack versions are targeted, while others are untargeted. For targeted attacks, AdvSecureNet either allows the user to specify the target label (DPatch) - or uses an automated target generation mechanism, choosing either second most-likely label (as the first most likely is the true label) - TOG mislabeling most-likely -- or the least-likely label - TOG mislabeling least-likely. This automated TOG mechanism ensures that the attack is targeted, as the goal is to mislead the model into predicting the target label instead of the correct one without being explicitly specified by the user. This feature is particularly useful for the large datasets where manually specifying target labels for each input image is impractical. However, for DPatch attacks users can also specify the target label manually if they would like to do so.

Below, we provide a brief overview of each adversarial attack supported by AdvSecureNet, including its characteristics, purpose, and potential applications.

## Adversarial Attacks Overview

Similarly as for image clasification, object detection adversarial attacks can be categorized in different ways. One way to categorize them is based on the **information** they use. Broadly, these attacks fall into two categories: **white-box** and **black-box** attacks[^1][^2]. White-box attacks necessitate access to the model's parameters, making them intrinsically reliant on detailed knowledge of the model's internals[^1][^2]. In contrast, black-box attacks operate without requiring access to the model's parameters[^1][^2]. Both of the aforementioned attacks are white-box attacks.

Another classification of adversarial attacks hinges on the **objective of the attack**. In this context, attacks are grouped into **targeted** and **untargeted** categories[^1]. Targeted attacks are designed with the specific goal of manipulating the model's output to a predetermined class. In contrast, untargeted attacks are aimed at causing the model to incorrectly classify the input into any class, provided it is not the correct one[^1].

Last but not least, object detection attacks can be split based on the **method of perturbation generation**. Attacks can fall into two categories: **pixel perturbation attacks** and **adversarial patch generation attacks**. Pixel perturbation attacks impact the whole image by crafting small changes to the input which are invisible to humans. They show high success rate and low perceptibility, however they are not robust to real-world conditions. Adversarial patch generation attacks, on the other hand, focus on generating a localized image - often can be thought of as a colorful "sticker", which, when placed anywhere in the scene, can fool object detectors. They do not require modifying the entire image and are transferrable to real world scenarios - however, they are less stealthy than pixel-level attacks and may require large-sized patches in order to work.

---

### DPatch

DPatch is a type of adversarial patch generation attack introduced by Xin Liu et al.[^3] in 2019. It is a white-box attack and can be used both as a targeted and untargeted attack. The idea is to:
- either find a patch pattern $\hat{P}_{u}$ that maximizes the loss L to true class label $\hat y$ and bounding box label $\hat B$ when the patch is applied - **DPatch untargeted**:
    $$
    \hat{P}_{u}
    = \arg\max_{P}
      \; \mathbb{E}_{x,s}\!\bigl[
        L\bigl(A(x, s, P);\,\hat y, \hat B\bigr)
      \bigr]
    $$
- or find a patch pattern $P_t$ that minimizes the loss $L$ to targeted label class $y_t$ and bounding box label $B_t$ - **DPatch targeted**:
    $$
    \hat{P}_{t}
    = \arg\min_{P}
      \; \mathbb{E}_{x,s}\!\bigl[
        L\bigl(A(x, s, P);\,y_{t}, B_{t}\bigr)
      \bigr]
    $$
where:
- $A$ - function of applying patch $P$ onto input scene $x$ with shift $s$. 

The goal of DPatch attack is to train the adversarial patch pattern that once attached to the input image, RoIs extracted by the detector gather in the region where the patch is attached. It is an iterative attack which can be applied in real-world scenarios. DPatch attack is one of the first adversarial patch based attacks for object detection and it is often used as a benchmark or reference point for evaluating new adversarial attacks in this space. What makes DPatch especially interesting is its transferability - it has been shown to be effective among different detectors and training datasets. 

---

### TOG

TOG is a type of pixel perturbation attack introduced by Ka-Ho Chow et al.[^4] in 2020. It is a white-box attack and has plenty of versions: untargeted, mislabeling, fabrication - only to name a few. The main goals of the attack are to:

- **TOG untargeted** - perturb each pixel in any way which increases the loss the most - as a result, the object detector will randomly misdetect and give incorrect results (of any form): 
$$
x_t' = \Pi_{x, \epsilon} \left[ x_{t-1}' + \alpha_{\mathrm{TOG}} \Gamma \left( \frac{\partial \mathcal{L}(x_{t-1}'; \hat{\mathcal{O}}(x), \mathbf{W})}{\partial x_{t-1}'} \right) \right],
$$
- **TOG vanishing** - make sure no candidate survives thresholding - as a result, the object detector will return no detections:
$$
x_t' = \Pi_{x, \epsilon} \left[ x_{t-1}' - \alpha_{\mathrm{TOG}} \Gamma \left( \frac{\partial \mathcal{L}_{\mathrm{obj}}(x_{t-1}'; \emptyset, \mathbf{W})}{\partial x_{t-1}'} \right) \right].
$$
- **TOG fabrication** - add false positives (additional detections) by maximizing the object existences using gradients from the objectness loss function $L_{obj}$:
$$
x_t' = \Pi_{x, \epsilon} \left[ x_{t-1}' + \alpha_{\mathrm{TOG}} \Gamma \left( \frac{\partial \mathcal{L}_{\mathrm{obj}}(x_{t-1}'; \emptyset, \mathbf{W})}{\partial x_{t-1}'} \right) \right].
$$
- **TOG mislabeling** - cause the object detector to consistently missclassify the detected objects on the input image by replacing their source label with the chosen target class label:
$$
x_t' = \Pi_{x, \epsilon} \left[ x_{t-1}' - \alpha_{\mathrm{TOG}} \Gamma \left( \frac{\partial \mathcal{L}(x_{t-1}'; \mathcal{O}^*(x), \mathbf{W})}{\partial x_{t-1}'} \right) \right].$$
where:
- $x$ - input image
- $\Pi_{x, \epsilon}$ - projection onto a hypersphere with a radius $\epsilon$ centered at $x$ in $L_p$ norm, followed by clipping to ensure the validity of pixel values
- $\alpha _{TOG}$ - attack learning rate
- $\Gamma$ - sign function
- $\mathbf{W}$ - learnable model weights
- $\mathcal{O}$ - set of ground truth objects
- $\mathcal{O}^*$ - auxiliary target detections - constructed by setting each object $\hat o $ in $\hat{\mathcal{O}}$ to its malicious class label
- $\mathcal{L}$ - loss function, consisting of objectness loss $\mathcal{L}_{obj}$, bounding box loss $\mathcal{L}_{bbox}$, and classification loss $\mathcal{L}_{class}$.

TOG attack is considered a backbone of more up to date pixel perturbation attacks and is often used as a benchmark for other object detection attacks.

---

### References

[^1]: Khalid, F., Hanif, M. A., & Shafique, M. (2021). Exploiting Vulnerabilities in Deep Neural Networks: Adversarial and Fault-Injection Attacks. arXiv preprint arXiv:2105.03251.
[^2]: Chakraborty, A., Alam, M., Dey, V., Chattopadhyay, A., & Mukhopadhyay, D. (2018). Adversarial Attacks and Defences: A Survey. arXiv preprint arXiv:1810.00069.
[^2]: Xin Liu and Huanrui Yang and Ziwei Liu and Linghao Song and Hai Li and Yiran Chen (2019). DPatch: An Adversarial Patch Attack on Object Detectors. arXiv preprint arXiv:1806.02299.
[^4]: Chow, Ka-Ho and Liu, Ling and Loper, Margaret and Bae, Juhyun and Gursoy, Mehmet Emre and Truex, Stacey and Wei, Wenqi and Wu, Yanzhao (2020). Adversarial Objectness Gradient Attacks in Real-time Object Detection Systems. 2020 Second IEEE International Conference on Trust, Privacy and Security in Intelligent Systems and Applications (TPS-ISA), pages 263-272.
