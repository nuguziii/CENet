import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # state
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--auto_resume', action='store_true')

    # path
    parser.add_argument('--data_dir',
                        help='dataset directory',
                        type=str,
                        default='E:\INFINITT\dataset')
    parser.add_argument('--output_dir',
                        help='output directory',
                        type=str,
                        default='C:\\Users\CGIP\Desktop\github\Deeply-Self-Supervised-Contour-Embedded-NN\\results')
    parser.add_argument('--description',
                        help='current train/test description',
                        type=str,
                        default='t1')
    parser.add_argument('--log_dir',
                        help='log directory',
                        type=str,
                        default='C:\\Users\CGIP\Desktop\github\Deeply-Self-Supervised-Contour-Embedded-NN\log')
    parser.add_argument('--model_name',
                        help='model name',
                        type=str,
                        default='model_best.pth')

    # network
    parser.add_argument('--network', type=str, default='CENet')

    # parameter
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=300)

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_args()
    if opt.train:
        from train import train
        train(opt)
    elif opt.test:
        from test import test
        test(opt)

if __name__ == '__main__':
    main()