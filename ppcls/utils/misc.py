# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import paddle

__all__ = ['AverageMeter']


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Code was based on https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, name='', fmt='f', postfix="", need_avg=True):
        self.name = name
        self.fmt = fmt
        self.postfix = postfix
        self.need_avg = need_avg
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    @property
    def avg_info(self):
        if isinstance(self.avg, paddle.Tensor):
            self.avg = float(self.avg)
        return "{}: {:.5f}".format(self.name, self.avg)

    @property
    def total(self):
        return '{self.name}_sum: {self.sum:{self.fmt}}{self.postfix}'.format(
            self=self)

    @property
    def total_minute(self):
        return '{self.name} {s:{self.fmt}}{self.postfix} min'.format(
            s=self.sum / 60, self=self)

    @property
    def mean(self):
        return '{self.name}: {self.avg:{self.fmt}}{self.postfix}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        return '{self.name}: {self.val:{self.fmt}}{self.postfix}'.format(
            self=self)



class AttrMeter(object):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.attr_names = {
            'color': ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black"],
            'type': ["sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck", "estate"],
            'brand': ["Others", "Honda", "Mazda", "Mitsubishi", "Suzuki", "Toyota", "Hyundai", "KIA", "VinFast","Chevrolet","Ford"],
            'model':["Others",
              "Honda City", "Honda Civic", "Honda Accord", "Honda Odyssey",
              "Mazda 3",  "Mazda 6", "Mazda cx3", "Mazda cx5", "Mazda cx9",
              "Mitsubishi Attrage",  "Mitsubishi Lancer",  "Mitsubishi Pajero",
              "Suzuki Swift", "Suzuki Wagonr",
              "Toyota Vios", "Toyota Avalon", "Toyota Camery", "Toyota Corolla", "Toyota fj", "Toyota Fortuner", "Toyota Hiace",
              "Toyota Hilux", "Toyota Innova", "Toyota Prado", "Toyota Rav4", "Toyota Yaris",
              "Hyundai Accent", "Hyundai H1", "Hyundai Santafe", "Hyundai Sonata",
              "KIA Cadenza", "KIA Cerato",  "KIA Optima", "KIA Picanto",  "KIA Rio", "KIA Sportage",
              "VinFast E34", "VinFast Fadil", "VinFast VF3", "VinFast VF5-9",
              "Chevrolet Aveo", "Chevrolet Impala",  "Chevrolet Malibu", "Chevrolet Silverado",  "Chevrolet Tahoe", "Chevrolet Traverse",
              "Ford Edge", "Ford Expedition",  "Ford Explorer", "Ford F150",  "Ford Flex", "Ford Ranger", "Taurus"
              ]
        }
        self.reset()

    def reset(self):
        """Initialize/reset all metrics"""
        # Per-class metrics
        for idx in range(84):  # Total 28 classes
            setattr(self, f'class_{idx}_gt_pos', 0)
            setattr(self, f'class_{idx}_gt_neg', 0)
            setattr(self, f'class_{idx}_true_pos', 0)
            setattr(self, f'class_{idx}_true_neg', 0)
            setattr(self, f'class_{idx}_false_pos', 0)
            setattr(self, f'class_{idx}_false_neg', 0)

        # Overall metrics
        self.overall_gt_pos = 0
        self.overall_gt_neg = 0
        self.overall_true_pos = 0
        self.overall_true_neg = 0
        self.overall_false_pos = 0
        self.overall_false_neg = 0

        # Instance-level metrics
        self.gt_pos_ins = []
        self.true_pos_ins = []
        self.intersect_pos = []
        self.union_pos = []
    def update(self, metric_dict):
            """Update metrics with new batch results"""
            # Update per-class metrics
            for idx in range(84):
                for metric_type in ['gt_pos', 'gt_neg', 'true_pos', 'true_neg', 'false_pos', 'false_neg']:
                    class_metric = f'class_{idx}_{metric_type}'
                    if class_metric in metric_dict:
                        curr_val = getattr(self, class_metric)
                        setattr(self, class_metric, curr_val + metric_dict[class_metric])

            # Update overall metrics
            self.overall_gt_pos += np.sum(metric_dict['gt_pos'])
            self.overall_gt_neg += np.sum(metric_dict['gt_neg'])
            self.overall_true_pos += np.sum(metric_dict['true_pos'])
            self.overall_true_neg += np.sum(metric_dict['true_neg'])
            self.overall_false_pos += np.sum(metric_dict['false_pos'])
            self.overall_false_neg += np.sum(metric_dict['false_neg'])

            # Update instance-level metrics
            if 'gt_pos_ins' in metric_dict:
                self.gt_pos_ins.extend(metric_dict['gt_pos_ins'].tolist())
                self.true_pos_ins.extend(metric_dict['true_pos_ins'].tolist())
                self.intersect_pos.extend(metric_dict['intersect_pos'].tolist())
                self.union_pos.extend(metric_dict['union_pos'].tolist())
    def calculate_per_class_metrics(self, class_idx):
        """Calculate metrics for a specific class with its name"""
        eps = 1e-20
        
        # Get class name based on index
        if class_idx < 10:
            class_name = self.attr_names['color'][class_idx]
            group = 'Color'
        elif class_idx < 19:
            class_name = self.attr_names['type'][class_idx - 10]
            group = 'Type'
        elif class_idx < 30:
            class_name = self.attr_names['brand'][class_idx - 19]
            group = 'Brand'
        else:
            class_name = self.attr_names['model'][class_idx - 30]
            group = 'Model'

        # Get metrics
        gt_pos = getattr(self, f'class_{class_idx}_gt_pos')
        gt_neg = getattr(self, f'class_{class_idx}_gt_neg')
        true_pos = getattr(self, f'class_{class_idx}_true_pos')
        true_neg = getattr(self, f'class_{class_idx}_true_neg')
        false_pos = getattr(self, f'class_{class_idx}_false_pos')
        false_neg = getattr(self, f'class_{class_idx}_false_neg')

        accuracy = (true_pos + true_neg) / (gt_pos + gt_neg + eps)
        precision = true_pos / (true_pos + false_pos + eps)
        recall = true_pos / (gt_pos + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return {
            'name': class_name,
            'group': group,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def calculate_group_metrics(self, group):
        """Calculate metrics for a specific group (overall, color, type, brand)"""
        eps = 1e-20

        if group == 'overall':
            gt_pos = self.overall_gt_pos
            gt_neg = self.overall_gt_neg
            true_pos = self.overall_true_pos
            true_neg = self.overall_true_neg
            false_pos = self.overall_false_pos
            false_neg = self.overall_false_neg
        else:
            group_indices = {
                'color': (0, 10),
                'type': (10, 19),
                'brand': (19, 30),
                'model': (30, 84)  # Giả sử model có tổng cộng 44 loại
            }

            # Lấy khoảng chỉ số tương ứng với nhóm hiện tại
            start_idx, end_idx = group_indices.get(group, (0, 0))

            # Tính tổng các chỉ số cho tất cả các lớp trong nhóm
            gt_pos = sum(getattr(self, f'class_{i}_gt_pos') for i in range(start_idx, end_idx))
            gt_neg = sum(getattr(self, f'class_{i}_gt_neg') for i in range(start_idx, end_idx))
            true_pos = sum(getattr(self, f'class_{i}_true_pos') for i in range(start_idx, end_idx))
            true_neg = sum(getattr(self, f'class_{i}_true_neg') for i in range(start_idx, end_idx))
            false_pos = sum(getattr(self, f'class_{i}_false_pos') for i in range(start_idx, end_idx))
            false_neg = sum(getattr(self, f'class_{i}_false_neg') for i in range(start_idx, end_idx))

        # Calculate metrics
        accuracy = (true_pos + true_neg) / (gt_pos + gt_neg + eps)
        precision = true_pos / (true_pos + false_pos + eps)
        recall = true_pos / (gt_pos + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def res(self):
      def format_metrics(name, metrics, indent=0):
          pad = " " * indent
          return (f"{pad}{name}\n"
                  f"{pad}{'=' * len(name)}\n"
                  f"{pad}Accuracy : {metrics['accuracy']:.2%}\n"
                  f"{pad}F1-Score : {metrics['f1']:.2%}\n"
                  f"{pad}Precision: {metrics['precision']:.2%}\n"
                  f"{pad}Recall   : {metrics['recall']:.2%}\n")

      output = []
      
      # Overall Performance Section
      overall = self.calculate_group_metrics('overall')
      output.append("\n" + "="*50)
      output.append("VEHICLE ATTRIBUTE CLASSIFICATION RESULTS")
      output.append("="*50 + "\n")
      output.append(format_metrics("OVERALL PERFORMANCE", overall))

      # Group Sections
      groups = {
          'color': ("COLOR ATTRIBUTES", 0, 10),
          'type': ("VEHICLE TYPES", 10, 19),
          'brand': ("BRAND CLASSIFICATION", 19, 30),
          'model': ("MODEL CLASSIFICATION", 30, 84)
      }

      for group_name, (title, start_idx, end_idx) in groups.items():
          # Group metrics
          group_metrics = self.calculate_group_metrics(group_name)
          output.append("\n" + "-"*50)
          output.append(format_metrics(title, group_metrics))
          
          # Individual class metrics
          output.append("\nDetailed Class Performance:")
          output.append("-" * 25)
          for idx in range(start_idx, end_idx):
              metrics = self.calculate_per_class_metrics(idx)
              class_name = metrics['name'].upper()
              acc = metrics['accuracy']
              f1 = metrics['f1']
              prec = metrics['precision']
              rec = metrics['recall']
              
              # Only show classes with non-zero metrics
              # if f1 > 0:
              output.append(f"Class: {class_name:<20} | "
                          f"Acc: {acc:>6.1%} | "
                          f"F1: {f1:>6.1%} | "
                          f"Prec: {prec:>6.1%} | "
                          f"Rec: {rec:>6.1%}")

      return "\n".join(output)
