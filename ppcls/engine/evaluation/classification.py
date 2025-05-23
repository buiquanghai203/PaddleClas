# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import platform
import paddle

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger


def classification_eval(engine, epoch_id=0):
    if hasattr(engine.eval_metric_func, "reset"):
        engine.eval_metric_func.reset()
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]

    tic = time.time()
    accum_samples = 0
    total_samples = len(
        engine.eval_dataloader.
        dataset) if not engine.use_dali else engine.eval_dataloader.size
    max_iter = len(engine.eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(engine.eval_dataloader)
    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()

        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        batch[0] = paddle.to_tensor(batch[0])
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([-1, 1]).astype("int64")

        # image input
        with engine.auto_cast(is_eval=True):
            if engine.is_rec:
                out = engine.model(batch[0], batch[1])
            else:
                out = engine.model(batch[0])

        # just for DistributedBatchSampler issue: repeat sampling
        current_samples = batch_size * paddle.distributed.get_world_size()
        accum_samples += current_samples

        if isinstance(out, dict) and "Student" in out:
            out = out["Student"]
        if isinstance(out, dict) and "logits" in out:
            out = out["logits"]

        # gather Tensor when distributed
        if paddle.distributed.get_world_size() > 1:
            label_list = []
            device_id = paddle.distributed.ParallelEnv().device_id
            label = batch[1].cuda(device_id) if engine.config["Global"][
                "device"] == "gpu" else batch[1]
            paddle.distributed.all_gather(label_list, label)
            labels = paddle.concat(label_list, 0)

            if isinstance(out, list):
                preds = []
                for x in out:
                    pred_list = []
                    paddle.distributed.all_gather(pred_list, x)
                    pred_x = paddle.concat(pred_list, 0)
                    preds.append(pred_x)
            else:
                pred_list = []
                paddle.distributed.all_gather(pred_list, out)
                preds = paddle.concat(pred_list, 0)

            if accum_samples > total_samples and not engine.use_dali:
                if isinstance(preds, list):
                    preds = [
                        pred[:total_samples + current_samples - accum_samples]
                        for pred in preds
                    ]
                else:
                    preds = preds[:total_samples + current_samples -
                                  accum_samples]
                labels = labels[:total_samples + current_samples -
                                accum_samples]
                current_samples = total_samples + current_samples - accum_samples
        else:
            labels = batch[1]
            preds = out

        # calc loss
        if engine.eval_loss_func is not None:
            with engine.auto_cast(is_eval=True):
                loss_dict = engine.eval_loss_func(preds, labels)

            for key in loss_dict:
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')
                output_info[key].update(float(loss_dict[key]), current_samples)

        #  calc metric
        if engine.eval_metric_func is not None:
            engine.eval_metric_func(preds, labels)
        time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            if "ATTRMetric" in engine.config["Metric"]["Eval"][0]:
                metric_msg = ""
            else:
                metric_msg = ", ".join([
                    "{}: {:.5f}".format(key, output_info[key].val)
                    for key in output_info
                ])
                if "MultiLabelMAP" not in engine.config["Metric"]["Eval"][0]:
                    metric_msg += ", {}".format(engine.eval_metric_func.avg_info)
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()
    if engine.use_dali:
        engine.eval_dataloader.reset()

    
    if "ATTRMetric" in engine.config["Metric"]["Eval"][0]:
        # Get AttrMeter instance
        attr_meter = engine.eval_metric_func.metric_func_list[0].attrmeter
        
        # Get detailed results
        eval_results = attr_meter.res()
        
        

        # Calculate mean accuracy (ma)
        eps = 1e-20
        true_pos = attr_meter.overall_true_pos
        true_neg = attr_meter.overall_true_neg
        gt_pos = attr_meter.overall_gt_pos
        gt_neg = attr_meter.overall_gt_neg
        
        # Calculate positive and negative recall
        label_pos_recall = true_pos / (gt_pos + eps)
        label_neg_recall = true_neg / (gt_neg + eps)
        
       
        ma = (label_pos_recall + label_neg_recall) / 2
        
        # Log ma value
        logger.info(f"[Eval][Epoch {epoch_id}] MA: {ma:.4f}")
        # Log the detailed results
        logger.info("[Eval][Epoch {}] Detailed Metrics:".format(epoch_id))
        logger.info(eval_results)
        return ma
    else:
        metric_msg = ", ".join([
            "{}: {:.5f}".format(key, output_info[key].avg)
            for key in output_info
        ])
        metric_msg += ", {}".format(engine.eval_metric_func.avg_info)
        logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

        # do not try to save best eval.model
        if engine.eval_metric_func is None:
            return -1
        # return 1st metric in the dict
        return engine.eval_metric_func.avg
