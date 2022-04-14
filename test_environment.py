import os

import cv2

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
        self.NTHUDDD_path = "images/Fold3_part2/"
        self.CEW_path = "images/dataset_B_FacialImages/"
        self.D3S_open_close_path = "images/D3S_dataset/Sub1/"

    def run_test(self, alg_to_test, input_video, labels=None):  
        predictions = alg_to_test.inference_on_video(input_video)
        performance_metrics, failed_predictions = self.compute_predictions_quality(predictions, labels)
        return { "predictions": predictions, "performance_metrics": performance_metrics }

    def test_open_close_eye_detection(self, alg_to_test):
        images, labels, filenames = self.prepare_CEW_dataset()

        predictions_dict = {}
        performance_metrics_dict = {}

        if False:
            predictions = []
            ind = 0
            for image in images:
    
                result = alg_to_test.process_frame(image)
                
                if result is not None:
                    predictions.append(result["open_eyes"])
                else:
                    predictions.append(None)
                
                ind += 1

            performance_metrics, failed_predictions = self.compute_predictions_quality(predictions, labels)
            predictions_dict["CEW"] = predictions
            performance_metrics_dict["CEW"] = performance_metrics

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
        open_eyes_dir = os.path.join(self.CEW_path, "OpenFace")
        for filename in os.listdir(open_eyes_dir):
            file = os.path.join(open_eyes_dir, filename)
            if os.path.isfile(file) and ".jpg" in filename:
                image = cv2.imread(file)
                images.append(image)
                labels.append(True)
                filenames.append(file)

        closed_eyes_dir = os.path.join(self.CEW_path, "ClosedFace")
        for filename in os.listdir(closed_eyes_dir):
            file = os.path.join(closed_eyes_dir, filename)
            if os.path.isfile(file) and ".jpg" in filename:
                image = cv2.imread(file)
                images.append(image)
                labels.append(False)
                filenames.append(file)
        
        return images, labels, filenames

    def prepare_NTHUDDD_dataset(self):
        video_dict = get_all_videos_from_directory(self.NTHUDDD_path)
        
        for subject, videos in video_dict.items():
            print(subject)
            for label, video in videos.items():
                print(label)
                print(video)

        return video_dict, video_dict

    def compute_predictions_quality(self, predictions, labels) -> dict:
        # TODO: compute accuracy, sensitivity, ...
        performance_metrics = {}   
        
        failed_predictions = []

        num_frames = len(predictions)
        num_hits = 0
        for i in range(0, num_frames):
            if predictions[i] == labels[i]:
                num_hits += 1
            else:
                failed_predictions.append(i)

        performance_metrics["accuracy"] = num_hits / num_frames
        return performance_metrics, failed_predictions