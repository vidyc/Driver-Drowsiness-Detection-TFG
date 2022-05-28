from cProfile import label
from cmath import nan
import os
from re import sub
import math

import cv2
import pandas as pd
import numpy as np

import inference as inf
import metrics_obtention as mo
import image_analysis

def get_all_videos_from_directory(directory: str, videos=None):
    videos = {}
    for filename in os.listdir(directory):
        file = os.path.join(directory, filename)

        if os.path.isdir(file):
            videos[filename] = {}
            for filename2 in os.listdir(file):
                video_path = os.path.join(file, filename2)
                video = cv2.VideoCapture(video_path)
                videos[filename][filename2] = video

    return videos

class TestEnvironment:

    def __init__(self):
        self.max_num_frames = 20
        self.NTHUDDD_training_path = "images/NTHUDDD_dataset/Training_Evaluation_Dataset/Training Dataset/"
        self.NTHUDDD_test_path = "images/NTHUDDD_dataset/Testing_Dataset/"
        self.eyeblink8_path = "images/eyeblink8/"
        self.CEW_path = "images/dataset_B_FacialImages/"
        self.D3S_open_close_path = "images/D3S_dataset/Sub1/"
        self.YAWNDD_path = "images/YAWNDD_dataset/"

    def run_test(self, alg_to_test, input_video, labels=None):  
        predictions = alg_to_test.inference_on_video(input_video)
        performance_metrics, failed_predictions = self.compute_predictions_quality(predictions, labels)
        return { "predictions": predictions, "performance_metrics": performance_metrics }

    def test_open_close_eye_detection(self, source_df, config):
        method = config["eye_closure_method"]
        threshold = config["eye_closure_threshold"]
        performance_metrics = {"CEW": {}}
    
        ear_info = np.array(source_df[f"ear{method}"])
        labels = np.array(source_df["label"])

        predictions = []
        for ear_value in ear_info:
            open_eye = ear_value >= threshold
            predictions.append(open_eye)

        num_hits = 0
        num_samples = len(labels)
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for ind, eye_state in enumerate(predictions):
            label = labels[ind]

            if eye_state == label:
                num_hits += 1
                if eye_state:
                    true_positives += 1
                else:
                    true_negatives += 1
            else:
                if label:
                    false_negatives += 1
                else:
                    false_positives += 1

        performance_metrics["CEW"]["true_positives"] = true_positives
        performance_metrics["CEW"]["true_negatives"] = true_negatives
        performance_metrics["CEW"]["false_positives"] = false_positives
        performance_metrics["CEW"]["false_negatives"] = false_negatives

        performance_metrics["CEW"]["accuracy"] = num_hits / num_samples
        if (true_positives + false_positives) != 0:
            performance_metrics["CEW"]["precision"] = true_positives / (true_positives + false_positives)
        else:
            performance_metrics["CEW"]["precision"] = None

        if (true_positives + false_negatives) != 0:
            performance_metrics["CEW"]["recall"] = true_positives / (true_positives + false_negatives)
        else:
            performance_metrics["CEW"]["recall"] = None

        if performance_metrics["CEW"]["recall"] is not None and performance_metrics["CEW"]["precision"] is not None:
            recall = performance_metrics["CEW"]["recall"]
            precision = performance_metrics["CEW"]["precision"]
            performance_metrics["CEW"]["f1-score"] = 2 * (recall * precision) / (recall + precision)

        # performance_metrics, failed_predictions = self.compute_predictions_quality(predictions, labels)
        # predictions_dict["CEW"] = predictions
        # performance_metrics_dict["CEW"] = performance_metrics

        return performance_metrics

    def test_open_close_eye_detection_videos(self, alg_to_test, features, videos, config, num_frames=None, video_names=[]):
        num_videos = len(videos)

        if video_names == []:
            for i in range(0, num_videos):
                video_names.append(f"output{i}.avi")

        for ind, video in enumerate(videos):
            if num_frames is not None:
                inf.inference_on_video(video, alg_to_test, features, config, video_names[ind], max_num_frames=num_frames)
            else:
                inf.inference_on_video(video, alg_to_test, features, config, video_names[ind])


    def prepare_yawn_dataset(self, root_folder):
        labels = []
        images = []
        filenames = []
        yawn_dir = os.path.join(root_folder, "yawn")
        for filename in os.listdir(yawn_dir):
            file = os.path.join(yawn_dir, filename)
            if os.path.isfile(file) and ".jpg" in filename:
                image = cv2.imread(file)
                images.append(image)
                labels.append(True)
                filenames.append(file)

        no_yawn_dir = os.path.join(root_folder, "no_yawn")
        for filename in os.listdir(no_yawn_dir):
            file = os.path.join(no_yawn_dir, filename)
            if os.path.isfile(file) and ".jpg" in filename:
                image = cv2.imread(file)
                images.append(image)
                labels.append(False)
                filenames.append(file)
        
        return images, labels, filenames

    def test_yawn_detection(self, alg_to_test):
        images, labels, filenames = self.prepare_yawn_dataset(self.YAWNDD_path)
        
        predictions = []
        ind = 0
        for image in images:
  
            result = alg_to_test.process_frame(image)
            
            if result is not None:
                predictions.append(result["yawn"])
            else:
                predictions.append(None)
            
            ind += 1

        performance_metrics, failed_predictions = self.compute_predictions_quality(predictions, labels)
        print(failed_predictions)
        return { "predictions": predictions, "performance_metrics": performance_metrics }

    def prepare_CEW_dataset(self):
        labels = []
        images = []
        filenames = []
        open_eyes_dir = os.path.join(self.CEW_path, "Open")
        for filename in os.listdir(open_eyes_dir):
            file = os.path.join(open_eyes_dir, filename)
            if os.path.isfile(file) and ".jpg" in filename:
                image = cv2.imread(file)
                images.append(image)
                labels.append(True)
                filenames.append(file)

        closed_eyes_dir = os.path.join(self.CEW_path, "Closed")
        for filename in os.listdir(closed_eyes_dir):
            file = os.path.join(closed_eyes_dir, filename)
            if os.path.isfile(file) and ".jpg" in filename:
                image = cv2.imread(file)
                images.append(image)
                labels.append(False)
                filenames.append(file)
        
        return images, labels, filenames


    def test_open_close_eye_detection_eyeblink8(self, source_df, config):
        videos_and_labels = self.get_videos_and_labels_eyeblink8()
        method = config["eye_closure_method"]
        threshold = config["eye_closure_threshold"]
        threshold_perc = config["main_threshold_perc"]
        gray_zone_perc = config["gray_zone_perc"]

        performance_metrics = {"total": {}}
        num_total_hits = 0
        num_total_samples = 0
        num_total_closed_frames = 0
        num_total_open_frames = 0
        num_total_true_positives = 0
        num_total_true_negatives = 0
        num_total_false_positives = 0
        num_total_false_negatives = 0
        for subject, data in videos_and_labels.items():
            subject_df = source_df[source_df["subject"] == subject]
            performance_metrics[subject] = {}
            labels = data["labels"]
            ear_info = np.array(subject_df[f"ear{method}"])
            previous_eye_state = True
            predictions = {}
            last_ear_values = []
            for frame_number, ear_value in enumerate(ear_info):
                if math.isnan(ear_value):
                    continue

                if last_ear_values != []:
                    threshold = threshold_perc * sum(last_ear_values) / len(last_ear_values)

                current_eye_state = image_analysis.check_eyes_open(ear_value, threshold, gray_zone_perc, previous_eye_state)
                predictions[frame_number] = current_eye_state
                previous_eye_state = current_eye_state
                
                if len(last_ear_values) >= config["num_frames_dynamic_avg"]:
                    last_ear_values.pop(0)
                last_ear_values.append(ear_value)
            
            num_hits = 0
            num_samples = len(labels)
            num_total_samples += num_samples
            num_closed_frames = len([x for x in labels.values() if not x])
            num_open_frames = len([x for x in labels.values() if x])
            num_total_closed_frames += num_closed_frames
            num_total_open_frames += num_open_frames
            num_true_positives = 0
            num_true_negatives = 0
            num_false_positives = 0
            num_false_negatives = 0

            for ind, eye_state in predictions.items():
                if ind not in labels:
                    continue

                label = labels[ind]

                if eye_state == label:
                    num_hits += 1
                    num_total_hits += 1
                    if eye_state:
                        num_true_positives += 1
                        num_total_true_positives += 1
                    else:
                        num_true_negatives += 1
                        num_total_true_negatives += 1
                else:
                    if label:
                        num_false_negatives += 1
                        num_total_false_negatives += 1
                    else:
                        num_false_positives += 1
                        num_total_false_positives += 1

            performance_metrics[subject]["num_open"] = num_open_frames
            performance_metrics[subject]["num_closed"] = num_closed_frames
            performance_metrics[subject]["true_positives"] = num_true_positives
            performance_metrics[subject]["true_negatives"] = num_true_negatives
            performance_metrics[subject]["false_positives"] = num_false_positives
            performance_metrics[subject]["false_negatives"] = num_false_negatives

            performance_metrics[subject]["accuracy"] = num_hits / num_samples
            
            if (num_true_positives + num_false_positives) != 0:
                performance_metrics[subject]["precision"] = num_true_positives / (num_true_positives + num_false_positives)
            else:
                performance_metrics[subject]["precision"] = None

            if (num_true_positives + num_false_negatives) != 0:
                performance_metrics[subject]["recall"] = num_true_positives / (num_true_positives + num_false_negatives)
            else:
                performance_metrics[subject]["recall"] = None
            
            if (num_false_positives + num_total_true_negatives) != 0:
                performance_metrics[subject]["specificity"] = num_true_negatives / (num_true_negatives + num_false_positives)
            else:
                performance_metrics[subject]["specificity"] = None

            if performance_metrics[subject]["recall"] is not None and performance_metrics[subject]["precision"] is not None:
                recall = performance_metrics[subject]["recall"]
                precision = performance_metrics[subject]["precision"]
                performance_metrics[subject]["f1-score"] = 2 * (recall * precision) / (recall + precision)
            
            if performance_metrics[subject]["recall"] is not None and performance_metrics[subject]["specificity"] is not None:
                recall = performance_metrics[subject]["recall"]
                specificity = performance_metrics[subject]["specificity"]
                performance_metrics[subject]["g-mean"] = math.sqrt(recall * specificity)
        
        performance_metrics["total"]["num_open"] = num_total_open_frames
        performance_metrics["total"]["num_closed"] = num_total_closed_frames
        performance_metrics["total"]["true_positives"] = num_total_true_positives
        performance_metrics["total"]["true_negatives"] = num_total_true_negatives
        performance_metrics["total"]["false_positives"] = num_total_false_positives
        performance_metrics["total"]["false_negatives"] = num_total_false_negatives

        print(f"Num open frames: {num_total_open_frames}")
        print(f"Num closed frames: {num_total_closed_frames}")
        print(f"True positives: {num_total_true_positives}")
        print(f"True negatives: {num_total_true_negatives}")
        print(f"False positives: {num_total_false_positives}")
        print(f"False negatives: {num_total_false_negatives}")

        performance_metrics["total"]["accuracy"] = num_total_hits / num_total_samples
        if (num_total_true_positives + num_total_false_positives) != 0:
            performance_metrics["total"]["precision"] = num_total_true_positives / (num_total_true_positives + num_total_false_positives)
        else:
            performance_metrics["total"]["precision"] = None

        if (num_total_true_positives + num_total_false_negatives) != 0:
            performance_metrics["total"]["recall"] = num_total_true_positives / (num_total_true_positives + num_total_false_negatives)
        else:
            performance_metrics["total"]["recall"] = None

        if (num_total_false_positives + num_total_true_negatives) != 0:
            performance_metrics["total"]["specificity"] = num_total_true_negatives / (num_total_true_negatives + num_total_false_positives)
        else:
            performance_metrics["total"]["specificity"] = None

        if performance_metrics["total"]["recall"] is not None and performance_metrics["total"]["precision"] is not None:
            recall = performance_metrics["total"]["recall"]
            precision = performance_metrics["total"]["precision"]
            performance_metrics["total"]["f1-score"] = 2 * (recall * precision) / (recall + precision)

        if performance_metrics["total"]["recall"] is not None and performance_metrics["total"]["specificity"] is not None:
            recall = performance_metrics["total"]["recall"]
            specificity = performance_metrics["total"]["specificity"]
            performance_metrics["total"]["g-mean"] = math.sqrt(recall * specificity)
        return performance_metrics

    def test_blink_detection_eyeblink8(self, source_df, config):
        videos_and_labels = self.get_videos_and_labels_eyeblink8()
        method = config["eye_closure_method"]
        threshold = config["eye_closure_threshold"]
        threshold_perc = config["main_threshold_perc"]
        gray_zone_perc = config["gray_zone_perc"]

        performance_metrics = {"total": {}}
        num_total_hits = 0
        num_total_real_closed_eyes = 0
        num_total_method_closed_eyes = 0
        num_total_hit_closed_eyes = 0
        num_total_real_blinks = 0
        num_total_method_blinks = 0
        num_total_true_positives = 0
        num_total_false_positives = 0
        num_total_false_negatives = 0
        for subject, data in videos_and_labels.items():
            subject_df = source_df[source_df["subject"] == subject]
            performance_metrics[subject] = {}
            blink_intervals = data["blink_intervals"]
            ear_info = np.array(subject_df[f"ear{method}"])
            previous_eye_state = True
            blink_set = []
            method_blink_intervals = []
            last_ear_values = []
            for frame_number, ear_value in enumerate(ear_info):
                if math.isnan(ear_value):
                    continue

                if last_ear_values != []:
                    threshold = threshold_perc * sum(last_ear_values) / len(last_ear_values)

                current_eye_state = image_analysis.check_eyes_open(ear_value, threshold, gray_zone_perc, previous_eye_state)
                if not current_eye_state:
                    blink_set.append(frame_number)
                
                if current_eye_state and not previous_eye_state:
                    method_blink_intervals.append(blink_set)
                    blink_set = []
                previous_eye_state = current_eye_state
                
                if len(last_ear_values) >= config["num_frames_dynamic_avg"]:
                    last_ear_values.pop(0)
                last_ear_values.append(ear_value)
            
            real_num_blinks = len(blink_intervals)
            num_total_real_blinks += real_num_blinks
            method_num_blinks = len(method_blink_intervals)
            num_total_method_blinks += method_num_blinks
            num_hits = 0
            num_true_positives = 0
            num_false_positives = 0
            num_false_negatives = 0

            ind_real = 0
            ind_method = 0

            while ind_method < method_num_blinks and ind_real < real_num_blinks:
                real_interval = blink_intervals[ind_real]
                method_interval = method_blink_intervals[ind_method]
                
                if set(method_interval).intersection(set(real_interval)):
                    num_hits += 1
                    num_total_hits += 1
                    num_true_positives += 1
                    num_total_true_positives += 1
                    ind_method += 1
                    ind_real += 1
                elif method_interval[-1] > real_interval[-1]:
                    num_false_negatives += 1
                    num_total_false_negatives += 1
                    ind_real += 1
                else:
                    num_false_positives += 1
                    num_total_false_positives += 1
                    ind_method += 1
            
            for i in range(ind_real, real_num_blinks):
                num_false_negatives += 1
                num_total_false_negatives += 1

            for i in range(ind_method, method_num_blinks):
                num_false_positives += 1
                num_total_false_positives += 1

            num_real_closed_eyes = sum([len(interval) for interval in blink_intervals])
            num_method_closed_eyes = sum([len(interval) for interval in method_blink_intervals])

            ind_real_closed_eyes = [ ind for interval in blink_intervals for ind in interval ]
            ind_method_closed_eyes = [ ind for interval in method_blink_intervals for ind in interval ]

            num_hit_closed_eyes = len(set(ind_method_closed_eyes).intersection(set(ind_real_closed_eyes)))
            num_total_hit_closed_eyes += num_hit_closed_eyes
            num_total_real_closed_eyes += num_real_closed_eyes
            num_total_method_closed_eyes += num_method_closed_eyes
            if subject == 1:
            #     print(blink_intervals)
            #     print(method_blink_intervals)
                print(f"num closed eyes: {sum([len(interval) for interval in blink_intervals])}")
                print(f"num predicted closed eyes: {sum([len(interval) for interval in method_blink_intervals])}")
            
            performance_metrics[subject]["num_real_blinks"] = real_num_blinks
            performance_metrics[subject]["num_method_blinks"] = method_num_blinks
            performance_metrics[subject]["num_real_closed_eyes"] = num_real_closed_eyes
            performance_metrics[subject]["num_method_closed_eyes"] = num_method_closed_eyes
            performance_metrics[subject]["num_hit_closed_eyes"] = num_hit_closed_eyes
            performance_metrics[subject]["true_positives"] = num_true_positives
            performance_metrics[subject]["false_positives"] = num_false_positives
            performance_metrics[subject]["false_negatives"] = num_false_negatives

            performance_metrics[subject]["accuracy"] = num_hits / real_num_blinks
            
            if (num_true_positives + num_false_positives) != 0:
                performance_metrics[subject]["precision"] = num_true_positives / (num_true_positives + num_false_positives)
            else:
                performance_metrics[subject]["precision"] = None

            if (num_true_positives + num_false_negatives) != 0:
                performance_metrics[subject]["recall"] = num_true_positives / (num_true_positives + num_false_negatives)
            else:
                performance_metrics[subject]["recall"] = None
            
            if performance_metrics[subject]["recall"] is not None and performance_metrics[subject]["precision"] is not None:
                recall = performance_metrics[subject]["recall"]
                precision = performance_metrics[subject]["precision"]
                performance_metrics[subject]["f1-score"] = 2 * (recall * precision) / (recall + precision)
                    
        performance_metrics["total"]["num_real_blinks"] = num_total_real_blinks
        performance_metrics["total"]["num_method_blinks"] = num_total_method_blinks
        performance_metrics["total"]["num_real_closed_eyes"] = num_total_real_closed_eyes
        performance_metrics["total"]["num_method_closed_eyes"] = num_total_method_closed_eyes
        performance_metrics["total"]["num_hit_closed_eyes"] = num_total_hit_closed_eyes
        performance_metrics["total"]["true_positives"] = num_total_true_positives
        performance_metrics["total"]["false_positives"] = num_total_false_positives
        performance_metrics["total"]["false_negatives"] = num_total_false_negatives

        print(f"Total real blinks: {num_total_real_blinks}")
        print(f"Total method blinks: {num_total_method_blinks}")
        print(f"True positives: {num_total_true_positives}")
        print(f"False positives: {num_total_false_positives}")
        print(f"False negatives: {num_total_false_negatives}")

        performance_metrics["total"]["accuracy"] = num_total_hits / num_total_real_blinks
        if (num_total_true_positives + num_total_false_positives) != 0:
            performance_metrics["total"]["precision"] = num_total_true_positives / (num_total_true_positives + num_total_false_positives)
        else:
            performance_metrics["total"]["precision"] = None

        if (num_total_true_positives + num_total_false_negatives) != 0:
            performance_metrics["total"]["recall"] = num_total_true_positives / (num_total_true_positives + num_total_false_negatives)
        else:
            performance_metrics["total"]["recall"] = None

        if performance_metrics["total"]["recall"] is not None and performance_metrics["total"]["precision"] is not None:
            recall = performance_metrics["total"]["recall"]
            precision = performance_metrics["total"]["precision"]
            performance_metrics["total"]["f1-score"] = 2 * (recall * precision) / (recall + precision)
        return performance_metrics

    
    def test_yawn_detection_NTHUDDD(self, source_df, config):
        method = config["mouth_closure_method"]
        threshold = config["mouth_yawn_threshold"]
        gray_zone = config["gray_zone_mouth"]
        num_past_frames = config["num_past_frames_mouth"]

        performance_metrics = {"total": {}}
        num_total_hits = 0
        num_total_real_open_mouth = 0
        num_total_method_open_mouth = 0
        num_total_hit_open_mouth = 0
        num_total_real_yawns = 0
        num_total_method_yawns = 0
        num_total_true_positives = 0
        num_total_true_negatives = 0
        num_total_false_positives = 0
        num_total_false_negatives = 0

        id_list = list(source_df["id"].unique())
        for id in id_list:
            
            id_df = source_df[source_df["id"] == id]
            performance_metrics[id] = {}
            
            frame_numbers = list(id_df["frame_count"] - 1)
            labels = list(id_df["mouth"])
            yawn_intervals = []
            yawn_set = []
            previous_label = 0
            # print(frame_numbers)
            for ind, label in enumerate(labels):
                frame_number = frame_numbers[ind]

                # 0 = nada, 1 = bostezo, 2 = hablar o reirse
                if label == 1:
                    yawn_set.append(frame_number)
                
                if label != 1 and previous_label == 1:
                    yawn_intervals.append(yawn_set)
                    yawn_set = []

                previous_label = label

            mar_info = np.array(id_df[f"mar{method}"])
            previous_yawn_state = None
            yawn_set = []
            method_yawn_intervals = []
            last_mar_values = []
            for frame_number, mar_value in enumerate(mar_info):
                if math.isnan(mar_value):
                    continue
                
                mean_mar = mar_value
                if last_mar_values != []:
                    mean_mar = (sum(last_mar_values) + mar_value) / (len(last_mar_values) + 1)

                upper_threshold = threshold + gray_zone
                lower_threshold = threshold - gray_zone
                if mean_mar < lower_threshold:
                    current_yawn_state = False
                elif mean_mar > upper_threshold:
                    current_yawn_state = True
                else:
                    current_yawn_state = mean_mar >= threshold
                    if previous_yawn_state is not None:
                        current_yawn_state = previous_yawn_state
                # current_yawn_state = mean_mar >= threshold

                if current_yawn_state:
                    yawn_set.append(frame_number)
                
                if not current_yawn_state and previous_yawn_state:
                    method_yawn_intervals.append(yawn_set)
                    yawn_set = []
                previous_yawn_state = current_yawn_state
                
                if num_past_frames > 0:
                    if len(last_mar_values) >= num_past_frames:
                        last_mar_values.pop(0)
                    last_mar_values.append(mar_value)
            
            real_num_yawns = len(yawn_intervals)
            num_total_real_yawns += real_num_yawns
            method_num_yawns = len(method_yawn_intervals)
            num_total_method_yawns += method_num_yawns
            num_hits = 0
            num_true_positives = 0
            num_true_negatives = 0
            num_false_positives = 0
            num_false_negatives = 0

            ind_real = 0
            ind_method = 0

            if id == "034noglassesyawning":
                print(f"THRESHOLD: {threshold}")
                print(f"REAL_INTS: {yawn_intervals}")
                print(f"METHOD_INTS: {method_yawn_intervals}")

            while ind_method < method_num_yawns and ind_real < real_num_yawns:
                real_interval = yawn_intervals[ind_real]
                method_interval = method_yawn_intervals[ind_method]
                
                if set(method_interval).intersection(set(real_interval)):
                    num_hits += 1
                    num_total_hits += 1
                    num_true_positives += 1
                    num_total_true_positives += 1
                    ind_method += 1
                    ind_real += 1
                elif method_interval[-1] > real_interval[-1]:
                    num_false_negatives += 1
                    num_total_false_negatives += 1
                    ind_real += 1
                else:
                    num_false_positives += 1
                    num_total_false_positives += 1
                    ind_method += 1
            
            for i in range(ind_real, real_num_yawns):
                num_false_negatives += 1
                num_total_false_negatives += 1

            for i in range(ind_method, method_num_yawns):
                num_false_positives += 1
                num_total_false_positives += 1

            num_real_open_mouth = sum([len(interval) for interval in yawn_intervals])
            num_method_open_mouth = sum([len(interval) for interval in method_yawn_intervals])

            ind_real_open_mouth = [ ind for interval in yawn_intervals for ind in interval ]
            ind_method_open_mouth = [ ind for interval in method_yawn_intervals for ind in interval ]

            num_hit_open_mouth = len(set(ind_method_open_mouth).intersection(set(ind_real_open_mouth)))
            num_total_hit_open_mouth += num_hit_open_mouth
            num_total_real_open_mouth += num_real_open_mouth
            num_total_method_open_mouth += num_method_open_mouth
            if id == "001noglassessleepy":
            #     print(yawn_intervals)
            #     print(method_yawn_intervals)
                print(f"num open mouth: {sum([len(interval) for interval in yawn_intervals])}")
                print(f"num predicted open mouth: {sum([len(interval) for interval in method_yawn_intervals])}")
            
            performance_metrics[id]["num_real_yawns"] = real_num_yawns
            performance_metrics[id]["num_method_yawns"] = method_num_yawns
            performance_metrics[id]["num_real_open_mouth"] = num_real_open_mouth
            performance_metrics[id]["num_method_open_mouth"] = num_method_open_mouth
            performance_metrics[id]["num_hit_open_mouth"] = num_hit_open_mouth
            performance_metrics[id]["true_positives"] = num_true_positives
            performance_metrics[id]["false_positives"] = num_false_positives
            performance_metrics[id]["false_negatives"] = num_false_negatives

            if real_num_yawns > 0:
                performance_metrics[id]["accuracy"] = num_hits / real_num_yawns
            else:
                performance_metrics[id]["accuracy"] = None

            if (num_true_positives + num_false_positives) != 0:
                performance_metrics[id]["precision"] = num_true_positives / (num_true_positives + num_false_positives)
            else:
                performance_metrics[id]["precision"] = None

            if (num_true_positives + num_false_negatives) != 0:
                performance_metrics[id]["recall"] = num_true_positives / (num_true_positives + num_false_negatives)
            else:
                performance_metrics[id]["recall"] = None
            
            recall = performance_metrics[id]["recall"]
            precision = performance_metrics[id]["precision"]
            if recall is not None and precision is not None and (recall + precision) > 0:
                performance_metrics[id]["f1-score"] = 2 * (recall * precision) / (recall + precision)
            else:
                performance_metrics[id]["f1-score"] = None
            
        performance_metrics["total"]["num_real_yawns"] = num_total_real_yawns
        performance_metrics["total"]["num_method_yawns"] = num_total_method_yawns
        performance_metrics["total"]["num_real_open_mouth"] = num_total_real_open_mouth
        performance_metrics["total"]["num_method_open_mouth"] = num_total_method_open_mouth
        performance_metrics["total"]["num_hit_open_mouth"] = num_total_hit_open_mouth
        performance_metrics["total"]["true_positives"] = num_total_true_positives
        performance_metrics["total"]["false_positives"] = num_total_false_positives
        performance_metrics["total"]["false_negatives"] = num_total_false_negatives

        print(f"Total real yawns: {num_total_real_yawns}")
        print(f"Total method yawns: {num_total_method_yawns}")
        print(f"True positives: {num_total_true_positives}")
        print(f"False positives: {num_total_false_positives}")
        print(f"False negatives: {num_total_false_negatives}")

        performance_metrics["total"]["accuracy"] = num_total_hits / num_total_real_yawns
        if (num_total_true_positives + num_total_false_positives) != 0:
            performance_metrics["total"]["precision"] = num_total_true_positives / (num_total_true_positives + num_total_false_positives)
        else:
            performance_metrics["total"]["precision"] = None

        if (num_total_true_positives + num_total_false_negatives) != 0:
            performance_metrics["total"]["recall"] = num_total_true_positives / (num_total_true_positives + num_total_false_negatives)
        else:
            performance_metrics["total"]["recall"] = None

        if performance_metrics["total"]["recall"] is not None and performance_metrics["total"]["precision"] is not None:
            recall = performance_metrics["total"]["recall"]
            precision = performance_metrics["total"]["precision"]
            performance_metrics["total"]["f1-score"] = 2 * (recall * precision) / (recall + precision)
        return performance_metrics
    

    def test_headnod_detection_NTHUDDD(self, source_df, config):
        method = config["pitch_method"]
        threshold = config["head_nod_threshold"]
        threshold_perc = config["head_nod_threshold_perc"]
        # gray_zone = config["gray_zone_pitch"]
        # num_past_frames = config["num_past_frames_pitch"]

        performance_metrics = {"total": {}}
        num_total_hits = 0
        num_total_real_head_down = 0
        num_total_method_head_down = 0
        num_total_hit_head_down = 0
        num_total_real_headnods = 0
        num_total_method_headnods = 0
        num_total_true_positives = 0
        num_total_true_negatives = 0
        num_total_false_positives = 0
        num_total_false_negatives = 0

        id_list = list(source_df["id"].unique())
        for id in id_list:
            if "sunglasses" in id:
                continue
            
            id_df = source_df[source_df["id"] == id]
            processed_df = mo.obtain_metrics_from_df(id_df, config)
            head_nod_states = list(processed_df["head_nod"])

            performance_metrics[id] = {}
            
            frame_numbers = list(id_df["frame_count"] - 1)
            labels = list(id_df["head"])
            headnod_intervals = []
            headnod_set = []
            previous_label = 0
            # print(frame_numbers)
            for ind, label in enumerate(labels):
                frame_number = frame_numbers[ind]

                # 0 = nada, 1 = headnod, 2 = de lado
                if label == 1:
                    headnod_set.append(frame_number)
                
                if label != 1 and previous_label == 1:
                    headnod_intervals.append(headnod_set)
                    headnod_set = []

                previous_label = label

            
            # pitch_info = np.array(id_df[f"pitch{method}"])
            # nose_tip_info = np.array(id_df["nose_tip_y"])
            previous_headnod_state = None
            headnod_set = []
            method_headnod_intervals = []
            last_pitch_values = []
            nose_tip_values = []
            threshold_val = threshold
            for frame_number, current_headnod_state in enumerate(head_nod_states):
                if math.isnan(current_headnod_state):
                    continue
                
                # nose_tip = nose_tip_info[frame_number]
                # mean_pitch = pitch_value

                # possible_head_nod = True
                # if nose_tip_values != []:
                #     mean_nose_tip_y = sum(nose_tip_values) / len(nose_tip_values)
                #     nose_tip_y_threshold = mean_nose_tip_y + config["head_nod_y_min_threshold"]
                #     possible_head_nod = nose_tip >= nose_tip_y_threshold

                # if last_pitch_values != [] and num_past_frames > 0:
                #     threshold_val = threshold_perc * 100#sum(last_pitch_values) / len(last_pitch_values)
                    
                #     pitch_last = min(num_past_frames, len(last_pitch_values))
                #     pitch_values_to_analyze = last_pitch_values[-pitch_last:]
                #     mean_pitch = (sum(pitch_values_to_analyze) + pitch_value) / (pitch_last + 1)

                # upper_threshold = threshold_perc*100 + gray_zone
                # lower_threshold = threshold_perc*100 - gray_zone
                # if mean_pitch < lower_threshold:
                #     current_headnod_state = True
                # elif mean_pitch > upper_threshold:
                #     current_headnod_state = False
                # else:
                #     current_headnod_state = mean_pitch <= threshold_perc*100
                #     if previous_headnod_state is not None:
                #         current_headnod_state = previous_headnod_state
                # current_headnod_state = possible_head_nod and mean_pitch <= threshold_perc*100

                if current_headnod_state:
                    headnod_set.append(frame_number)
                
                if not current_headnod_state and previous_headnod_state:
                    method_headnod_intervals.append(headnod_set)
                    headnod_set = []
                previous_headnod_state = current_headnod_state
                
                # if len(last_pitch_values) >= 100:
                #     last_pitch_values.pop(0)
                # last_pitch_values.append(pitch_value)

                # if len(nose_tip_values) >= config["num_frames_nose_tip_y"]:
                #     nose_tip_values.pop(0)
                # nose_tip_values.append(nose_tip)
            
            real_num_headnods = len(headnod_intervals)
            num_total_real_headnods += real_num_headnods
            method_num_headnods = len(method_headnod_intervals)
            num_total_method_headnods += method_num_headnods
            num_hits = 0
            num_true_positives = 0
            num_true_negatives = 0
            num_false_positives = 0
            num_false_negatives = 0

            ind_real = 0
            ind_method = 0

            # if id == "033glassessleepy":
            #     print(f"THRESHOLD_PERC: {threshold_perc}")
            #     print(f"THRESHOLD: {threshold}")
            #     print(f"REAL_INTS: {headnod_intervals}")
            #     print(f"METHOD_INTS: {method_headnod_intervals}")

            while ind_method < len(method_headnod_intervals) and ind_real < real_num_headnods:
                real_interval = headnod_intervals[ind_real]
                method_interval = method_headnod_intervals[ind_method]
                
                if set(method_interval).intersection(set(real_interval)):
                    num_hits += 1
                    num_total_hits += 1
                    num_true_positives += 1
                    num_total_true_positives += 1
                    ind_method += 1
                    ind_real += 1
                elif method_interval[-1] > real_interval[-1]:
                    num_false_negatives += 1
                    num_total_false_negatives += 1
                    ind_real += 1
                else:
                    num_false_positives += 1
                    num_total_false_positives += 1
                    ind_method += 1
            
            for i in range(ind_real, real_num_headnods):
                num_false_negatives += 1
                num_total_false_negatives += 1

            for i in range(ind_method, method_num_headnods):
                num_false_positives += 1
                num_total_false_positives += 1

            num_real_head_down = sum([len(interval) for interval in headnod_intervals])
            num_method_head_down = sum([len(interval) for interval in method_headnod_intervals])

            ind_real_head_down = [ ind for interval in headnod_intervals for ind in interval ]
            ind_method_head_down = [ ind for interval in method_headnod_intervals for ind in interval ]

            num_hit_head_down = len(set(ind_method_head_down).intersection(set(ind_real_head_down)))
            num_total_hit_head_down += num_hit_head_down
            num_total_real_head_down += num_real_head_down
            num_total_method_head_down += num_method_head_down
            
            performance_metrics[id]["num_real_headnods"] = real_num_headnods
            performance_metrics[id]["num_method_headnods"] = method_num_headnods
            performance_metrics[id]["num_real_head_down"] = num_real_head_down
            performance_metrics[id]["num_method_head_down"] = num_method_head_down
            performance_metrics[id]["num_hit_head_down"] = num_hit_head_down
            performance_metrics[id]["true_positives"] = num_true_positives
            performance_metrics[id]["false_positives"] = num_false_positives
            performance_metrics[id]["false_negatives"] = num_false_negatives

            if real_num_headnods > 0:
                performance_metrics[id]["accuracy"] = num_hits / real_num_headnods
            else:
                performance_metrics[id]["accuracy"] = None

            if (num_true_positives + num_false_positives) != 0:
                performance_metrics[id]["precision"] = num_true_positives / (num_true_positives + num_false_positives)
            else:
                performance_metrics[id]["precision"] = None

            if (num_true_positives + num_false_negatives) != 0:
                performance_metrics[id]["recall"] = num_true_positives / (num_true_positives + num_false_negatives)
            else:
                performance_metrics[id]["recall"] = None
            
            recall = performance_metrics[id]["recall"]
            precision = performance_metrics[id]["precision"]
            if recall is not None and precision is not None and (recall + precision) > 0:
                performance_metrics[id]["f1-score"] = 2 * (recall * precision) / (recall + precision)
            else:
                performance_metrics[id]["f1-score"] = None
            
        performance_metrics["total"]["num_real_headnods"] = num_total_real_headnods
        performance_metrics["total"]["num_method_headnods"] = num_total_method_headnods
        performance_metrics["total"]["num_real_head_down"] = num_total_real_head_down
        performance_metrics["total"]["num_method_head_down"] = num_total_method_head_down
        performance_metrics["total"]["num_hit_head_down"] = num_total_hit_head_down
        performance_metrics["total"]["true_positives"] = num_total_true_positives
        performance_metrics["total"]["false_positives"] = num_total_false_positives
        performance_metrics["total"]["false_negatives"] = num_total_false_negatives

        print(f"Total real headnods: {num_total_real_headnods}")
        print(f"Total method headnods: {num_total_method_headnods}")
        print(f"True positives: {num_total_true_positives}")
        print(f"False positives: {num_total_false_positives}")
        print(f"False negatives: {num_total_false_negatives}")

        performance_metrics["total"]["accuracy"] = num_total_hits / num_total_real_headnods
        if (num_total_true_positives + num_total_false_positives) != 0:
            performance_metrics["total"]["precision"] = num_total_true_positives / (num_total_true_positives + num_total_false_positives)
        else:
            performance_metrics["total"]["precision"] = None

        if (num_total_true_positives + num_total_false_negatives) != 0:
            performance_metrics["total"]["recall"] = num_total_true_positives / (num_total_true_positives + num_total_false_negatives)
        else:
            performance_metrics["total"]["recall"] = None

        recall = performance_metrics["total"]["recall"]
        precision = performance_metrics["total"]["precision"]
        if recall is not None and precision is not None and (recall + precision) > 0:
            performance_metrics["total"]["f1-score"] = 2 * (recall * precision) / (recall + precision)
        else:
            performance_metrics["total"]["f1-score"] = None
        return performance_metrics


    def process_label_file_eyeblink8(self, file):
        with open(file, 'r') as f:
            lines = f.readlines()

        start = False
        ind = 0
        while not start:
            line = lines[ind]
            start = "#start" in line
            ind += 1

        labels = {}
        line_length = len(lines)
        previous_blink_state = -1
        blink_set = []
        blink_intervals = []
        for i in range(ind, line_length-1):
            tokens = lines[i].split(":")
            frame_number = int(tokens[0])
            eye_blink_state = int(tokens[1])
            
            if eye_blink_state == -1 and previous_blink_state != -1:
                blink_intervals.append(blink_set)
                blink_set = []
            
            if eye_blink_state != -1 and previous_blink_state != -1 and eye_blink_state != previous_blink_state:
                blink_intervals.append(blink_set)
                blink_set = []    
            
            if eye_blink_state != -1:
                blink_set.append(frame_number)

            labels[frame_number] = eye_blink_state == -1
            previous_blink_state = eye_blink_state
        
        return {"blink_intervals": blink_intervals, "labels": labels}

    def get_videos_and_labels_eyeblink8(self):
        video_dict = {}
        path = self.eyeblink8_path
        for subj in os.listdir(path):
            subject = int(subj)
            file = os.path.join(path, subj)

            if os.path.isdir(file):
                video_dict[subject] = {}
                for data in os.listdir(file):
                    data_file = os.path.join(file, data)
                    if data[-4:] == ".avi":
                        video_cap = cv2.VideoCapture(data_file)
                        video_dict[subject]["video"] = video_cap
                    elif data[-4:] == ".tag":
                        label_data = self.process_label_file_eyeblink8(data_file)
                        video_dict[subject]["blink_intervals"] = label_data["blink_intervals"]
                        video_dict[subject]["labels"] = label_data["labels"]
            
        return video_dict

    def process_label_file_NTHUDDD(self, file):
        with open(file, 'r') as f:
            labels = f.read()
        
        labels = [ int(char) for char in labels if char != "\n" ]
        return labels

    def get_videos_and_labels_NTHUDDD(self):
        video_dict = {}
        paths = {"train": self.NTHUDDD_training_path, "test": self.NTHUDDD_test_path}
        for dataset, path in paths.items():
            video_dict[dataset] = {}
            for subject in os.listdir(path):
                file = os.path.join(path, subject)

                if os.path.isdir(file):
                    video_dict[dataset][subject] = {}
                    for scenario in os.listdir(file):
                        scenario_dir = os.path.join(file, scenario)
                        video_dict[dataset][subject][scenario] = {
                            "nonsleepy": {},
                            "sleepy": {},
                            "slowBlinkWithNodding": {},
                            "yawning": {},
                            "mix": {},
                        }
                        if os.path.isdir(file):
                            if dataset == "train":
                                video_dict[dataset][subject][scenario]["nonsleepy"]["labels"] = {}
                                video_dict[dataset][subject][scenario]["sleepy"]["labels"] = {}
                                video_dict[dataset][subject][scenario]["slowBlinkWithNodding"]["labels"] = {}
                                video_dict[dataset][subject][scenario]["yawning"]["labels"] = {}
                            else:
                                video_dict[dataset][subject][scenario]["mix"]["labels"] = {}
                            for video in os.listdir(scenario_dir):
                                video_path = os.path.join(scenario_dir, video)
                                if video[-4:] == ".txt":
                                    labels = self.process_label_file_NTHUDDD(video_path)
                                    label_type = video.split("_")[2][:-4]
                                    if "nonsleepy" in video:
                                        video_dict[dataset][subject][scenario]["nonsleepy"]["labels"][label_type] = labels
                                    elif "sleepy" in video:
                                        video_dict[dataset][subject][scenario]["sleepy"]["labels"][label_type] = labels
                                    elif "slowBlinkWithNodding" in video:
                                        video_dict[dataset][subject][scenario]["slowBlinkWithNodding"]["labels"][label_type] = labels
                                    elif "yawning" in video:
                                        video_dict[dataset][subject][scenario]["yawning"]["labels"][label_type] = labels
                                    elif "mix" in video:
                                        video_dict[dataset][subject][scenario]["mix"]["labels"]["drowsiness"] = labels
                                else:
                                    video_cap = cv2.VideoCapture(video_path)
                                    if "nonsleepy" in video:
                                        video_dict[dataset][subject][scenario]["nonsleepy"]["video"] = video_cap
                                    elif "sleepy" in video:
                                        video_dict[dataset][subject][scenario]["sleepy"]["video"] = video_cap
                                    elif "slowBlinkWithNodding" in video:
                                        video_dict[dataset][subject][scenario]["slowBlinkWithNodding"]["video"] = video_cap
                                    elif "yawning" in video:
                                        video_dict[dataset][subject][scenario]["yawning"]["video"] = video_cap
                                    elif "mix" in video:
                                        video_dict[dataset][subject][scenario]["mix"]["video"] = video_cap

        return video_dict

    def compute_predictions_quality(self, predictions, labels, true_val=True) -> dict:
        # TODO: compute accuracy, sensitivity, ...
        performance_metrics = {}   
        
        failed_predictions = []

        num_frames = len(predictions)
        num_hits = 0
        num_true_positives = 0
        num_true_negatives = 0
        num_false_positives = 0
        num_false_negatives = 0
        for i in range(0, num_frames):
            if predictions[i] == labels[i]:
                num_hits += 1
                if labels[i] == true_val:
                    num_true_positives += 1
                else:
                    num_true_negatives += 1
            else:
                if labels[i] == true_val:
                    num_false_negatives += 1
                else:
                    num_false_positives += 1
                failed_predictions.append(i)

        performance_metrics["accuracy"] = num_hits / num_frames

        if (num_true_positives + num_false_positives) != 0:
            performance_metrics["precision"] = num_true_positives / (num_true_positives + num_false_positives)
        else:
            performance_metrics["precision"] = None

        if (num_true_positives + num_false_negatives) != 0:
            performance_metrics["recall"] = num_true_positives / (num_true_positives + num_false_negatives)
        else:
            performance_metrics["recall"] = None
        return performance_metrics, failed_predictions