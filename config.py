import argparse
import datetime


nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H')
result_dir = './result/{}'.format(nowTime)


def cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=bool, default=True,
                        help='use gpu, default True')
    parser.add_argument('--model_path', type=str, default='{}/model_'.format(result_dir),
                        help='Path to save model')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--output_size', type=float, default=2.0,
                        help='Output duration')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sampling rate')
    parser.add_argument('--length', type=float, default=2.0,
                        help='Duration of input audio')
    parser.add_argument('--loss', type=str, default="L1",
                        help="L1 or L2")
    parser.add_argument('--channels', type=int, default=1,
                        help="Input channel, mono or sterno, default mono")
    parser.add_argument('--h5_dir', type=str, default='H5/',
                        help="Path of hdf5 file")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--hold_step", type=int, default=20,
                        help="Epochs of hold step")
    parser.add_argument("--example_freq", type=int, default=200,
                        help="write an audio summary into Tensorboard logs")
    return parser.parse_args()