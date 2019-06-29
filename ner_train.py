#!/usr//bin/env python3

from get_bert_estimator import get_estimator
import argparse
import time
import datetime as dt
import functools
import tensorflow as tf


def parse(serialized_example):
    """
    Function adapts features for the Estimator format
    :param serialized_example:
    :return:
    """
    features_spec = {
        'input_ids': tf.VarLenFeature(tf.int64),
        'input_masks': tf.VarLenFeature(tf.int64),
        'y_masks': tf.VarLenFeature(tf.int64),
        'labels': tf.VarLenFeature(tf.int64),
        'text': tf.VarLenFeature(tf.string),
    }

    example = tf.parse_single_example(serialized_example, features_spec)

    return ({
                'input_ids': tf.sparse.to_dense(example['input_ids']),
                'input_masks': tf.sparse.to_dense(example['input_masks']),
                'y_masks': tf.sparse.to_dense(example['y_masks']),
                'text': example['text'],
            },
            tf.sparse.to_dense(example['labels']))


def input_fn(tfrecord_ds_path, batch_size):
    dataset = tf.data.TFRecordDataset([tfrecord_ds_path])\
        .map(lambda record: parse(record)).shuffle(100).batch(batch_size).repeat()

    return dataset


def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders
    https://www.tensorflow.org/guide/saved_model#prepare_serving_inputs
    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    # TODO adapt me
    input_ids_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_indices_ph')
    input_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='token_mask_ph')
    y_masks_ph = tf.placeholder(shape=(None, None), dtype=tf.int32, name='y_mask_ph')
    receiver_tensors = {
        'input_ids': input_ids_ph,
        'input_masks': input_masks_ph,
        'y_masks': y_masks_ph,
    }

    return tf.estimator.export.ServingInputReceiver(receiver_tensors, receiver_tensors)


def main(train_dataset_path, eval_dataset_path, batch_size):
    # load bert as Estimator:
    bert_ner_estimator = get_estimator()

    early_stopping = tf.estimator.stop_if_no_decrease_hook(
        bert_ner_estimator,
        metric_name='loss',
        max_steps_without_decrease=1000,
        min_steps=100)

    train_spec = tf.estimator.TrainSpec(
        input_fn = functools.partial(input_fn, tfrecord_ds_path=train_dataset_path, batch_size=batch_size),
        hooks=[early_stopping],
    )

    export = tf.estimator.BestExporter('exporter', serving_input_receiver_fn=serving_input_receiver_fn)

    eval_spec = tf.estimator.EvalSpec(
        input_fn = functools.partial(input_fn, tfrecord_ds_path=eval_dataset_path, batch_size=batch_size),
        steps=128,
        exporters=[export]
    )

    tf.estimator.train_and_evaluate(
        bert_ner_estimator,
        train_spec,
        eval_spec
    )


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', help='Path to TFRecord dataset for training', default='data/train.tfrecord', type=str)
    parser.add_argument('--eval_dataset', help='Path to TFRecord dataset for evaluation', default='data/valid.tfrecord', type=str)
    parser.add_argument('--batch_size', help='Size of Batch', default=8, type=int)
    args = parser.parse_args()
    main(args.train_dataset, args.eval_dataset, args.batch_size)
