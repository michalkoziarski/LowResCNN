import os
import datetime
import argparse
import models
import containers
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-network', choices=['AlexNet', 'VGGNet', 'ResNet'])
parser.add_argument('-scaling_factor', type=int)
args = vars(parser.parse_args())

if args.get('network') is None:
    networks = {'AlexNet': models.AlexNet(), 'VGGNet': models.VGGNet(), 'ResNet': models.ResNet()}
else:
    networks = {args.get('network'): eval('models.%s()' % args.get('network'))}

if args.get('scaling_factor') is None:
    scaling_factors = [1, 2, 3, 4, 5, 6, 7, 8]
else:
    scaling_factors = [args.get('scaling_factor')]

with tf.Session() as session:
    results = []

    for name, network in networks.items():
        network.load(session)

        for scaling_factor in scaling_factors:
            dataset = containers.ValidationSet(scaling_factor=scaling_factor)
            scores = network.score(session, dataset, dataset.labels, top_k=[1, 5])

            results.append([name, scaling_factor, np.round(scores[0], 6), np.round(scores[1], 6)])

    results_path = os.path.join(os.path.dirname(__file__), 'results')

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    output_file_name = 'low_resolution_%s.csv' % datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S.%f')

    df = pd.DataFrame(results, columns=['network', 'scaling', 'top1', 'top5'])
    df.to_csv(os.path.join(results_path, output_file_name), index=False)
