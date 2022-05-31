You can use this code to verify the perturbations of existing indiscriminate poisoning attacks are linear separable.

The [clean](https://drive.google.com/drive/folders/1NpXyJozirOSJ5bXBSeK7rtx9kBA6VttE) and [perturbed](https://drive.google.com/drive/folders/1OD54_gK6wnhyVwQGnHs7vIsKVOL-48zd) data of [NTGA](https://github.com/lionelmessi6410/ntga) can be downloaded directly. Note that you need to subtract clean data from the perturbed data to get perturbations.


For [DeepConfuse](https://github.com/kingfengji/DeepConfuse), [error-minimizing noise](https://github.com/HanxunH/Unlearnable-Examples), and [error-maximizing noise](https://github.com/lhfowl/adversarial_poisons), we need to manually generate perturbed data using their official implementations. We run their code and provide the results at [here](https://drive.google.com/file/d/1v9mAzowQ1GVxjTWZfhjsICLdGyLfiWOY/view?usp=sharing).

If you have downloaded 'x_train_cifar10_ntga_cnn_best.npy', 'x_train_cifar10.npy', and 'y_train_cifar10.npy', run the following command to check the accuracy of a linear model:

    python test_linear_separability.py --hidden_layers 0 --perturbed_x_path x_train_cifar10_ntga_cnn_best.npy --clean_x_path x_train_cifar10.npy  --label_path y_train_cifar10.npy


