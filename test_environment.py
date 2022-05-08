import os

import cv2
from setuptools import sic

import inference as inf
import metrics_obtention as mo

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
        self.NTHUDDD_path = "images/NTHUDDD_dataset/Training_Evaluation_Dataset/Training Dataset/"
        self.CEW_path = "images/dataset_B_FacialImages/"
        self.D3S_open_close_path = "images/D3S_dataset/Sub1/"
        self.YAWNDD_path = "images/YAWNDD_dataset/"

    def run_test(self, alg_to_test, input_video, labels=None):  
        predictions = alg_to_test.inference_on_video(input_video)
        performance_metrics, failed_predictions = self.compute_predictions_quality(predictions, labels)
        return { "predictions": predictions, "performance_metrics": performance_metrics }

    def test_open_close_eye_detection(self):
        images, labels, filenames = self.prepare_CEW_dataset()

        predictions_dict = {}
        performance_metrics_dict = {}

        predictions = []
        ind = 0
        for image in images:

            result, _ = mo.process_frame(image)
            
            if result is not None:
                predictions.append(result["open_eyes"])
            else:
                predictions.append(None)
            
            ind += 1

        performance_metrics, failed_predictions = self.compute_predictions_quality(predictions, labels)
        predictions_dict["CEW"] = predictions
        performance_metrics_dict["CEW"] = performance_metrics

        if False:
            images, labels, filenames = self.prepare_D3S_open_close_dataset()
            
            predictions2 = []
            ind = 0
            for image in images:
    
                result = alg_to_test.process_frame(image)
                
                if result is not None:
                    predictions2.append(result["open_eyes"])
                else:
                    predictions2.append(None)
                
                ind += 1

            performance_metrics2, failed_predictions2 = self.compute_predictions_quality(predictions2, labels)
            predictions_dict["D3S"] = predictions2
            performance_metrics_dict["D3S"] = performance_metrics2
        return { "predictions": predictions_dict, "performance_metrics": performance_metrics_dict }

    def test_open_close_eye_detection_videos(self, alg_to_test, features, videos, config, num_frames=2000, video_names=[]):
        num_videos = len(videos)

        if video_names == []:
            for i in range(0, num_videos):
                video_names.append(f"output{i}.avi")

        for ind, video in enumerate(videos):
            inf.inference_on_video(video, alg_to_test, features, config, video_names[ind], max_num_frames=num_frames)


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

    def prepare_D3S_open_close_dataset(self):
        labels = []
        images = []
        filenames = []
        open_eyes_dir = os.path.join(self.D3S_open_close_path, "Open")
        for filename in os.listdir(open_eyes_dir):
            file = os.path.join(open_eyes_dir, filename)
            if os.path.isfile(file) and ".jpg" in filename:
                image = cv2.imread(file)
                images.append(image)
                labels.append(True)
                filenames.append(file)

        closed_eyes_dir = os.path.join(self.D3S_open_close_path, "Closed")
        for filename in os.listdir(closed_eyes_dir):
            file = os.path.join(closed_eyes_dir, filename)
            if os.path.isfile(file) and ".jpg" in filename:
                image = cv2.imread(file)
                images.append(image)
                labels.append(False)
                filenames.append(file)
        
        return images, labels, filenames

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

    def process_label_file_NTHUDDD(self, file):
        with open(file, 'r') as f:
            labels = f.read()
        
        labels = [ int(char) for char in labels ]
        return labels

    def get_videos_and_labels_NTHUDDD(self):
        video_dict = {}
        for subject in os.listdir(self.NTHUDDD_path):
            file = os.path.join(self.NTHUDDD_path, subject)

            if os.path.isdir(file):
                video_dict[subject] = {}
                for scenario in os.listdir(file):
                    scenario_dir = os.path.join(file, scenario)
                    video_dict[subject][scenario] = {
                        "nonsleepy": {},
                        "sleepy": {},
                        "slowBlinkWithNodding": {},
                        "yawning": {},
                    }
                    if os.path.isdir(file):
                        video_dict[subject][scenario]["nonsleepy"]["labels"] = {}
                        video_dict[subject][scenario]["sleepy"]["labels"] = {}
                        video_dict[subject][scenario]["slowBlinkWithNodding"]["labels"] = {}
                        video_dict[subject][scenario]["yawning"]["labels"] = {}
                        for video in os.listdir(scenario_dir):
                            video_path = os.path.join(scenario_dir, video)
                            if video[-4:] == ".txt":
                                labels = self.process_label_file_NTHUDDD(video_path)
                                label_type = video.split("_")[2][:-4]
                                if "nonsleepy" in video:
                                    video_dict[subject][scenario]["nonsleepy"]["labels"][label_type] = labels
                                elif "sleepy" in video:
                                    video_dict[subject][scenario]["sleepy"]["labels"][label_type] = labels
                                elif "slowBlinkWithNodding" in video:
                                    video_dict[subject][scenario]["slowBlinkWithNodding"]["labels"][label_type] = labels
                                elif "yawning" in video:
                                    video_dict[subject][scenario]["yawning"]["labels"][label_type] = labels
                            else:
                                video_cap = cv2.VideoCapture(video_path)
                                if "nonsleepy" in video:
                                    video_dict[subject][scenario]["nonsleepy"]["video"] = video_cap
                                elif "sleepy" in video:
                                    video_dict[subject][scenario]["sleepy"]["video"] = video_cap
                                elif "slowBlinkWithNodding" in video:
                                    video_dict[subject][scenario]["slowBlinkWithNodding"]["video"] = video_cap
                                elif "yawning" in video:
                                    video_dict[subject][scenario]["yawning"]["video"] = video_cap

        return video_dict


    def prepare_NTHUDDD_dataset(self):
        video_dict = get_all_videos_from_directory(self.NTHUDDD_path)
        
        # for subject, videos in video_dict.items():
        #     print(subject)
        #     for label, video in videos.items():
        #         print(label)
        #         print(video)

        return video_dict, video_dict

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