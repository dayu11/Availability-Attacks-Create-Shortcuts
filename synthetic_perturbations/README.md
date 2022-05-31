Here are some example commands to test synthetic shortcuts as availability attacks.

To test synthetic noises on CIFAR-10 and ResNet18 without data augmentation:

    CUDA_VISIBLE_DEVICES=0 python cifar_train.py --model resnet18 --dataset c10

To test synthetic noises on CIFAR-10 and ResNet18 with data augmentation:

    CUDA_VISIBLE_DEVICES=0 python cifar_train.py --model resnet18 --dataset c10 --aug

You can also change the model or dataset:

    CUDA_VISIBLE_DEVICES=0 python cifar_train.py --model densenet --dataset c100 --aug

Add '--clean' flag to train the model on clean data:

    CUDA_VISIBLE_DEVICES=0 python cifar_train.py --model resnet18 --dataset c10 --aug --clean
