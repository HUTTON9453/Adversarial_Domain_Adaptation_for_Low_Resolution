"""Main script for ADDA."""

import params
from core import eval_src, eval_tgt, train_src, train_tgt, eval_blur_tgt
from models import Discriminator, LeNetClassifier, LeNetEncoder, ResNet34, ResNet18, ResNetClassifier
from utils import get_data_loader, init_model, init_random_seed
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

if __name__ == '__main__':
    # init random seed
    init_random_seed(params.manual_seed)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    # load dataset
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
    dataset_original = CIFAR10(root=params.src_dataset, train=True, transform=transform_train)
    transform_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])
    dataset_original_val = CIFAR10(root=params.src_dataset, train=False, transform=transform_val)
    transform_train_blur = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.Resize((params.blur_size, params.blur_size)),
                                                transforms.Resize((32, 32)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean, std)])
    dataset_blur = CIFAR10(root=params.src_dataset, train=True, transform=transform_train_blur)
    transform_val_blur = transforms.Compose([transforms.Resize((params.blur_size, params.blur_size)),
                                            transforms.Resize((32, 32)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])
    dataset_blur_val = CIFAR10(root=params.src_dataset, train=False, transform=transform_val_blur)

    # src_data_loader = DataLoader(dataset_original, batch_size=params.batch_size, num_workers=4, shuffle=False, drop_last=True, pin_memory=True)
    # src_data_loader_eval = DataLoader(dataset_original_val, batch_size=params.batch_size, num_workers=4, pin_memory=True)
    # tgt_data_loader = DataLoader(dataset_blur, batch_size=params.batch_size, num_workers=4, shuffle=False, drop_last=True, pin_memory=True)
    # tgt_data_loader_eval = DataLoader(dataset_blur_val, batch_size=params.batch_size, num_workers=4, pin_memory=True)

    src_data_loader = DataLoader(dataset_blur, batch_size=params.batch_size, num_workers=4, shuffle=False, drop_last=True, pin_memory=True)
    src_data_loader_eval = DataLoader(dataset_blur_val, batch_size=params.batch_size, num_workers=4, pin_memory=True)

    # src_data_loader = get_data_loader(params.src_dataset)
    # src_data_loader_eval = get_data_loader(params.src_dataset, train=False)
    # tgt_data_loader = get_data_loader(params.tgt_dataset)
    # tgt_data_loader_eval = get_data_loader(params.tgt_dataset, train=False)

    # load models
    src_encoder = init_model(net=ResNet34(),
                             restore=params.src_encoder_restore)
    src_classifier = init_model(net=ResNetClassifier(),
                                restore=params.src_classifier_restore)
    blur_src_encoder = init_model(net=ResNet34(),
                             restore=params.blur_src_encoder_restore)
    blur_src_classifier = init_model(net=ResNetClassifier(),
                                restore=params.blur_src_classifier_restore)
    tgt_encoder = init_model(net=ResNet34(),
                             restore=params.tgt_encoder_restore)
    critic = init_model(Discriminator(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)

    if not (src_encoder.restored and src_classifier.restored and
            params.src_model_trained):
        src_encoder, src_classifier = train_src(
            src_encoder, src_classifier, src_data_loader)

    # eval source model
    print("=== Evaluating classifier for source domain ===")
    eval_src(src_encoder, src_classifier, src_data_loader_eval)
    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    eval_tgt(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> blur source only <<<")
    eval_tgt(blur_src_encoder, blur_src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    eval_tgt(tgt_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> blur domain adaption <<<")
    eval_blur_tgt(tgt_encoder, src_classifier, blur_src_encoder, blur_src_classifier, tgt_data_loader_eval, params.alpha)
