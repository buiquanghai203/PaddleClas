# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

import os
import copy
import shutil
from functools import partial
import importlib
import numpy
import numpy as np
import paddle
import paddle.nn.functional as F


def build_postprocess(config):
    if config is None:
        return None

    mod = importlib.import_module(__name__)
    config = copy.deepcopy(config)

    main_indicator = config.pop(
        "main_indicator") if "main_indicator" in config else None
    main_indicator = main_indicator if main_indicator else ""

    func_list = []
    for func in config:
        func_list.append(getattr(mod, func)(**config[func]))
    return PostProcesser(func_list, main_indicator)


def parse_class_id_map(class_id_map_file, delimiter):
    if class_id_map_file is None:
        return None

    if not os.path.exists(class_id_map_file):
        print(
            "Warning: If want to use your own label_dict, please input legal path!\nOtherwise label_names will be empty!"
        )
        return None

    try:
        class_id_map = {}
        with open(class_id_map_file, "r") as fin:
            lines = fin.readlines()
            for line in lines:
                partition = line.split("\n")[0].partition(delimiter)
                class_id_map[int(partition[0])] = str(partition[-1])
    except Exception as ex:
        print(ex)
        class_id_map = None
    return class_id_map


class PostProcesser(object):
    def __init__(self, func_list, main_indicator="Topk"):
        self.func_list = func_list
        self.main_indicator = main_indicator

    def __call__(self, x, image_file=None):
        rtn = None
        for func in self.func_list:
            tmp = func(x, image_file)
            if type(func).__name__ in self.main_indicator:
                rtn = tmp
        return rtn


class ThreshOutput(object):
    def __init__(self,
                 threshold=0,
                 default_label_index=0,
                 label_0="0",
                 label_1="1",
                 class_id_map_file=None,
                 delimiter=None):
        self.threshold = threshold
        self.default_label_index = default_label_index
        self.label_0 = label_0
        self.label_1 = label_1

        delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = parse_class_id_map(class_id_map_file, delimiter)

    def __call__(self, x, file_names=None):
        def binary_classification(x):
            y = []
            for idx, probs in enumerate(x):
                score = probs[1]
                if score < self.threshold:
                    result = {
                        "class_ids": [0],
                        "scores": [1 - score],
                        "label_names": [self.label_0]
                    }
                else:
                    result = {
                        "class_ids": [1],
                        "scores": [score],
                        "label_names": [self.label_1]
                    }
                if file_names is not None:
                    result["file_name"] = file_names[idx]
                y.append(result)
            return y

        def multi_classification(x):
            y = []
            for idx, probs in enumerate(x):
                index = probs.argsort(axis=0)[::-1].astype("int32")
                top1_id = index[0]
                top1_score = probs[top1_id]

                if top1_score > self.threshold:
                    rtn_id = top1_id
                else:
                    rtn_id = self.default_label_index

                label_name = self.class_id_map[
                    rtn_id] if self.class_id_map is not None else ""

                result = {
                    "class_ids": [rtn_id],
                    "scores": [probs[rtn_id]],
                    "label_names": [label_name]
                }
                if file_names is not None:
                    result["file_name"] = file_names[idx]
                y.append(result)
            return y

        if file_names is not None:
            assert x.shape[0] == len(file_names)
        if x.shape[1] == 2:
            return binary_classification(x)
        else:
            return multi_classification(x)


class ScoreOutput(object):
    def __init__(self, decimal_places):
        self.decimal_places = decimal_places

    def __call__(self, x, file_names=None):
        y = []
        for idx, probs in enumerate(x):
            score = np.around(x[idx], self.decimal_places)
            result = {"scores": score}
            if file_names is not None:
                result["file_name"] = file_names[idx]
            y.append(result)
        return y


class Topk(object):
    def __init__(self,
                 topk=1,
                 class_id_map_file=None,
                 delimiter=None,
                 label_list=None):
        assert isinstance(topk, (int, ))
        self.topk = topk
        delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = parse_class_id_map(
            class_id_map_file, delimiter) if not label_list else label_list

    def __call__(self, x, file_names=None):
        if file_names is not None:
            assert x.shape[0] == len(file_names)
        y = []
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-self.topk:][::-1].astype("int32")
            clas_id_list = []
            score_list = []
            label_name_list = []
            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    label_name_list.append(self.class_id_map[i.item()])
            result = {
                "class_ids": clas_id_list,
                "scores": np.around(
                    score_list, decimals=5).tolist(),
            }
            if file_names is not None:
                result["file_name"] = file_names[idx]
            if label_name_list is not None:
                result["label_names"] = label_name_list
            y.append(result)
        return y


class MultiLabelThreshOutput(object):
    def __init__(self, threshold=0.5, class_id_map_file=None, delimiter=None):
        self.threshold = threshold
        delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = parse_class_id_map(class_id_map_file, delimiter)

    def __call__(self, x, file_names=None):
        y = []
        for idx, probs in enumerate(x):
            index = np.where(probs >= self.threshold)[0].astype("int32")
            clas_id_list = []
            score_list = []
            label_name_list = []
            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    label_name_list.append(self.class_id_map[i.item()])
            result = {
                "class_ids": clas_id_list,
                "scores": np.around(
                    score_list, decimals=5).tolist(),
                "label_names": label_name_list
            }
            if file_names is not None:
                result["file_name"] = file_names[idx]
            y.append(result)
        return y


class SavePreLabel(object):
    def __init__(self, save_dir):
        if save_dir is None:
            raise Exception(
                "Please specify save_dir if SavePreLabel specified.")
        self.save_dir = partial(os.path.join, save_dir)

    def __call__(self, x, file_names=None):
        if file_names is None:
            return
        assert x.shape[0] == len(file_names)
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-1].astype("int32")
            self.save(index, file_names[idx])

    def save(self, id, image_file):
        output_dir = self.save_dir(str(id))
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(image_file, output_dir)


class Binarize(object):
    def __init__(self, method="round"):
        self.method = method
        self.unit = np.array([[128, 64, 32, 16, 8, 4, 2, 1]]).T

    def __call__(self, x, file_names=None):
        if self.method == "round":
            x = np.round(x + 1).astype("uint8") - 1

        if self.method == "sign":
            x = ((np.sign(x) + 1) / 2).astype("uint8")

        embedding_size = x.shape[1]
        assert embedding_size % 8 == 0, "The Binary index only support vectors with sizes multiple of 8"

        byte = np.zeros([x.shape[0], embedding_size // 8], dtype=np.uint8)
        for i in range(embedding_size // 8):
            byte[:, i:i + 1] = np.dot(x[:, i * 8:(i + 1) * 8], self.unit)

        return byte


class PersonAttribute(object):
    def __init__(self,
                 threshold=0.5,
                 glasses_threshold=0.3,
                 hold_threshold=0.6):
        self.threshold = threshold
        self.glasses_threshold = glasses_threshold
        self.hold_threshold = hold_threshold

    def __call__(self, batch_preds, file_names=None):
        # postprocess output of predictor
        age_list = ['AgeLess18', 'Age18-60', 'AgeOver60']
        direct_list = ['Front', 'Side', 'Back']
        bag_list = ['HandBag', 'ShoulderBag', 'Backpack']
        upper_list = ['UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice']
        lower_list = [
            'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts',
            'Skirt&Dress'
        ]
        batch_res = []
        for res in batch_preds:
            res = res.tolist()
            label_res = []
            # gender 
            gender = 'Female' if res[22] > self.threshold else 'Male'
            label_res.append(gender)
            # age
            age = age_list[np.argmax(res[19:22])]
            label_res.append(age)
            # direction 
            direction = direct_list[np.argmax(res[23:])]
            label_res.append(direction)
            # glasses
            glasses = 'Glasses: '
            if res[1] > self.glasses_threshold:
                glasses += 'True'
            else:
                glasses += 'False'
            label_res.append(glasses)
            # hat
            hat = 'Hat: '
            if res[0] > self.threshold:
                hat += 'True'
            else:
                hat += 'False'
            label_res.append(hat)
            # hold obj
            hold_obj = 'HoldObjectsInFront: '
            if res[18] > self.hold_threshold:
                hold_obj += 'True'
            else:
                hold_obj += 'False'
            label_res.append(hold_obj)
            # bag
            bag = bag_list[np.argmax(res[15:18])]
            bag_score = res[15 + np.argmax(res[15:18])]
            bag_label = bag if bag_score > self.threshold else 'No bag'
            label_res.append(bag_label)
            # upper
            upper_res = res[4:8]
            upper_label = 'Upper:'
            sleeve = 'LongSleeve' if res[3] > res[2] else 'ShortSleeve'
            upper_label += ' {}'.format(sleeve)
            for i, r in enumerate(upper_res):
                if r > self.threshold:
                    upper_label += ' {}'.format(upper_list[i])
            label_res.append(upper_label)
            # lower
            lower_res = res[8:14]
            lower_label = 'Lower: '
            has_lower = False
            for i, l in enumerate(lower_res):
                if l > self.threshold:
                    lower_label += ' {}'.format(lower_list[i])
                    has_lower = True
            if not has_lower:
                lower_label += ' {}'.format(lower_list[np.argmax(lower_res)])

            label_res.append(lower_label)
            # shoe
            shoe = 'Boots' if res[14] > self.threshold else 'No boots'
            label_res.append(shoe)

            threshold_list = [0.5] * len(res)
            threshold_list[1] = self.glasses_threshold
            threshold_list[18] = self.hold_threshold
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            batch_res.append({"attributes": label_res, "output": pred_res})
        return batch_res


class FaceAttribute(object):
    def __init__(self, threshold=0.65, convert_cn=False):
        self.threshold = threshold
        self.convert_cn = convert_cn

    def __call__(self, x, file_names=None):
        attribute_list = [
            ["CheekWhiskers", "刚长出的双颊胡须"], ["ArchedEyebrows", "柳叶眉"],
            ["Attractive", "吸引人的"], ["BagsUnderEyes", "眼袋"], ["Bald", "秃头"],
            ["Bangs", "刘海"], ["BigLips", "大嘴唇"], ["BigNose", "大鼻子"],
            ["BlackHair", "黑发"], ["BlondHair", "金发"], ["Blurry", "模糊的"],
            ["BrownHair", "棕发"], ["BushyEyebrows", "浓眉"], ["Chubby", "圆胖的"],
            ["DoubleChin", "双下巴"], ["Eyeglasses", "带眼镜"], ["Goatee", "山羊胡子"],
            ["GrayHair", "灰发或白发"], ["HeavyMakeup", "浓妆"],
            ["HighCheekbones", "高颧骨"], ["Male", "男性"],
            ["MouthSlightlyOpen", "微微张开嘴巴"], ["Mustache", "胡子"],
            ["NarrowEyes", "细长的眼睛"], ["NoBeard", "无胡子"],
            ["OvalFace", "椭圆形的脸"], ["PaleSkin", "苍白的皮肤"],
            ["PointyNose", "尖鼻子"], ["RecedingHairline", "发际线后移"],
            ["RosyCheeks", "红润的双颊"], ["Sideburns", "连鬓胡子"], ["Smiling", "微笑"],
            ["StraightHair", "直发"], ["WavyHair", "卷发"],
            ["WearingEarrings", "戴着耳环"], ["WearingHat", "戴着帽子"],
            ["WearingLipstick", "涂了唇膏"], ["WearingNecklace", "戴着项链"],
            ["WearingNecktie", "戴着领带"], ["Young", "年轻人"]
        ]
        gender_list = [["Male", "男性"], ["Female", "女性"]]
        age_list = [["Young", "年轻人"], ["Old", "老年人"]]
        batch_res = []
        index = 1 if self.convert_cn else 0
        for idx, res in enumerate(x):
            res = res.tolist()
            label_res = []
            threshold_list = [self.threshold] * len(res)
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            for i, value in enumerate(pred_res):
                if i == 20:
                    label_res.append(gender_list[0][index]
                                     if value == 1 else gender_list[1][index])
                elif i == 39:
                    label_res.append(age_list[0][index]
                                     if value == 1 else age_list[1][index])
                else:
                    if value == 1:
                        label_res.append(attribute_list[i][index])
            batch_res.append({"attributes": label_res, "output": pred_res})
        return batch_res


class VehicleAttribute(object):
    def __init__(self, color_threshold=0.5, type_threshold=0.5, brand_threshold=0.5, model_threshold=0.5):
        self.brand_threshold = brand_threshold
        self.color_threshold = color_threshold
        self.type_threshold = type_threshold
        self.model_threshold = model_threshold
        self.color_list = [
            "yellow", "orange", "green", "gray", "red", "blue", "white",
            "golden", "brown", "black"
        ]
        self.type_list = [
            "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus",
            "truck", "estate"
        ]
        self.brand_list = [
           "Others", "Honda", "Mazda", "Mitsubishi", "Suzuki", "Toyota", "Hyundai", "KIA", "VinFast", "Chevrolet", "Ford"
        ]
        self.model_list = [
           "Others",
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

    def __call__(self, x, file_names=None):
        device = paddle.get_device()
        if not isinstance(x, paddle.Tensor):
          x = paddle.to_tensor(x, place=device)
    
        # Process the output
        x = F.sigmoid(x)
        if 'gpu' in device:
            x = paddle.to_tensor(x, place=paddle.CPUPlace())
        x = x.numpy()

        # postprocess output of predictor
        batch_res = []
        for idx, res in enumerate(x):
            res = res.tolist()
            label_res = []
            color_idx = np.argmax(res[:10])
            type_idx = np.argmax(res[10:19])
            brand_idx = np.argmax(res[19:30])
            model_idx = np.argmax(res[30:])
            color_pred = [0] * 10
            type_pred = [0] * 9
            brand_pred = [0] * 11
            model_pred = [0] * 54
            #print(color_idx, type_idx, brand_idx)
            if res[color_idx] >= self.color_threshold:
                color_info = f"Color: ({self.color_list[color_idx]}, prob: {res[color_idx]})"
                color_pred[color_idx] = 1
            else:
                color_info = "Color unknown"
            if res[type_idx + 10] >= self.type_threshold:
                type_info = f"Type: ({self.type_list[type_idx]}, prob: {res[type_idx + 10]})"
                type_pred[type_idx] = 1
            else:
                type_info = "Type unknown"
            if res[brand_idx + 19] >= self.brand_threshold:
                brand_info = f"Brand: ({self.brand_list[brand_idx]}, prob: {res[brand_idx + 19]})"
                brand_pred[brand_idx] = 1
            else:
                brand_info = "Brand unknown"

            if res[model_idx + 30] >= self.model_threshold:
                model_info = f"Model: ({self.model_list[model_idx]}, prob: {res[model_idx + 30]})"
                model_pred[model_idx] = 1
            else:
                model_info = "Model unknown"
            label_res = f"{color_info}, {type_info}, {brand_info}, {model_info}"

            pred_res = color_pred + type_pred + brand_pred + model_pred
            batch_res.append({
                "attr": label_res,
                "pred": pred_res
              
            })
        return batch_res

class TableAttribute(object):
    def __init__(
            self,
            source_threshold=0.5,
            number_threshold=0.5,
            color_threshold=0.5,
            clarity_threshold=0.5,
            obstruction_threshold=0.5,
            angle_threshold=0.5, ):
        self.source_threshold = source_threshold
        self.number_threshold = number_threshold
        self.color_threshold = color_threshold
        self.clarity_threshold = clarity_threshold
        self.obstruction_threshold = obstruction_threshold
        self.angle_threshold = angle_threshold

    def __call__(self, batch_preds, file_names=None):
        # postprocess output of predictor
        batch_res = []

        for res in batch_preds:
            res = res.tolist()
            label_res = []
            source = 'Scanned' if res[0] > self.source_threshold else 'Photo'
            number = 'Little' if res[1] > self.number_threshold else 'Numerous'
            color = 'Black-and-White' if res[
                2] > self.color_threshold else 'Multicolor'
            clarity = 'Clear' if res[3] > self.clarity_threshold else 'Blurry'
            obstruction = 'Without-Obstacles' if res[
                4] > self.number_threshold else 'With-Obstacles'
            angle = 'Horizontal' if res[
                5] > self.number_threshold else 'Tilted'

            label_res = [source, number, color, clarity, obstruction, angle]

            threshold_list = [
                self.source_threshold, self.number_threshold,
                self.color_threshold, self.clarity_threshold,
                self.obstruction_threshold, self.angle_threshold
            ]
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            batch_res.append({"attributes": label_res, "output": pred_res})
        return batch_res
