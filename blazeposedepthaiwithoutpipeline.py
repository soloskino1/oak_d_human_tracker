import numpy as np
import mediapipe_utils as mpu
import cv2
from pathlib import Path
from FPS import FPS, now
from math import sin, cos
import depthai as dai
import time, sys
SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DETECTION_MODEL = str(SCRIPT_DIR / "models/pose_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/pose_landmark_full_sh4.blob")
LANDMARK_MODEL_HEAVY = str(SCRIPT_DIR / "models/pose_landmark_heavy_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/pose_landmark_lite_sh4.blob")
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()
class BlazeposeDepthai:
    def __init__(self, input_src="rgb",
                pd_model=None, 
                pd_score_thresh=0.5,
                lm_model=None,
                lm_score_thresh=0.7,
                xyz=False,
                crop=False,
                smoothing= True,
                internal_fps=None,
                resolution="full",
                internal_frame_height=1080,
                stats=False,
                trace=False,
                force_detection=False
                ):
        self.pd_model = pd_model if pd_model else POSE_DETECTION_MODEL
        print(f"Pose detection blob file : {self.pd_model}")
        self.rect_transf_scale = 1.25
        if lm_model is None or lm_model == "full":
            self.lm_model = LANDMARK_MODEL_FULL
        elif lm_model == "lite":
            self.lm_model = LANDMARK_MODEL_LITE
        elif lm_model == "heavy":
            self.lm_model = LANDMARK_MODEL_HEAVY
        else:
            self.lm_model = lm_model
        print(f"Landmarks using blob file : {self.lm_model}")
        self.pd_score_thresh = pd_score_thresh
        self.lm_score_thresh = lm_score_thresh
        self.smoothing = smoothing
        self.crop = crop 
        self.internal_fps = internal_fps     
        self.stats = stats
        self.force_detection = force_detection
        self.presence_threshold = 0.5
        self.visibility_threshold = 0.5
        self.device = dai.Device()
        self.xyz = False
        if input_src == None or input_src == "rgb" or input_src == "rgb_laconic":
            self.input_type = "rgb" 
            if internal_fps is None:
                if "heavy" in str(lm_model):
                    self.internal_fps = 10
                elif "full" in str(lm_model):
                    self.internal_fps = 8
                else: 
                    self.internal_fps = 13
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")
            if resolution == "full":
                self.resolution = (1920, 1080)
            elif resolution == "ultra":
                self.resolution = (3840, 2160)
            else:
                print(f"Error: {resolution} is not a valid resolution !")
                sys.exit()
            print("Sensor resolution:", self.resolution)
            self.video_fps = self.internal_fps 
            if xyz:
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                    self.xyz = True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")
            width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
            self.img_h = int(round(self.resolution[1] * self.scale_nd[0] / self.scale_nd[1]))
            self.img_w = int(round(self.resolution[0] * self.scale_nd[0] / self.scale_nd[1]))
            self.pad_h = (self.img_w - self.img_h) // 2
            self.pad_w = 0
            self.frame_size = self.img_w
            self.crop_w = 0
            print(f"Internal camera image size: {self.img_w} x {self.img_h} - crop_w:{self.crop_w} pad_h: {self.pad_h}")
        self.nb_kps = 33 
        if self.smoothing:
            self.filter_landmarks = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.05,
                beta=80,
                derivate_cutoff=1
            )
            self.filter_landmarks_aux = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.01,
                beta=10,
                derivate_cutoff=1
            )
            self.filter_landmarks_world = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.1,
                beta=40,
                derivate_cutoff=1,
                disable_value_scaling=True
            )
            if self.xyz:
                self.filter_xyz = mpu.LowPassFilter(alpha=0.25)
        self.anchors = mpu.generate_blazepose_anchors()
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")
        self.pd_input_length = 224
        self.lm_input_length = 256
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")
        if self.input_type == "rgb":
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            self.q_pre_pd_manip_cfg = self.device.getInputQueue(name="pre_pd_manip_cfg")
            if self.xyz:
                self.q_spatial_data = self.device.getOutputQueue(name="spatial_data_out", maxSize=1, blocking=False)
                self.q_spatial_config = self.device.getInputQueue("spatial_calc_config_in")
        else:
            self.q_pd_in = self.device.getInputQueue(name="pd_in")
        self.q_pd_out = self.device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
        self.q_lm_in = self.device.getInputQueue(name="lm_in")
        self.q_lm_out = self.device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
        self.fps = FPS()
        self.nb_frames = 0
        self.nb_pd_inferences = 0
        self.nb_lm_inferences = 0
        self.nb_lm_inferences_after_landmarks_ROI = 0
        self.nb_frames_no_body = 0
        self.glob_pd_rtrip_time = 0
        self.glob_lm_rtrip_time = 0
        self.use_previous_landmarks = False
        self.cfg_pre_pd = dai.ImageManipConfig()
        self.cfg_pre_pd.setResizeThumbnail(self.pd_input_length, self.pd_input_length)
    def is_present(self, body, lm_id):
        return body.presence[lm_id] > self.presence_threshold
    def is_visible(self, body, lm_id):
        if body.visibility[lm_id] > self.visibility_threshold and \
            0 <= body.landmarks[lm_id][0] < self.img_w and \
            0 <= body.landmarks[lm_id][1] < self.img_h :
            return True
        else:
            return False
    def query_body_xyz(self, body):
        if self.is_visible(body, mpu.KEYPOINT_DICT['right_hip']) and self.is_visible(body, mpu.KEYPOINT_DICT['left_hip']):
            body.xyz_ref = "mid_hips"
            body.xyz_ref_coords_pixel = np.mean([
                body.landmarks[mpu.KEYPOINT_DICT['right_hip']][:2],
                body.landmarks[mpu.KEYPOINT_DICT['left_hip']][:2]], 
                axis=0)
        elif self.is_visible(body, mpu.KEYPOINT_DICT['right_shoulder']) and self.is_visible(body, mpu.KEYPOINT_DICT['left_shoulder']):
            body.xyz_ref = "mid_shoulders"
            body.xyz_ref_coords_pixel = np.mean([
                body.landmarks[mpu.KEYPOINT_DICT['right_shoulder']][:2],
                body.landmarks[mpu.KEYPOINT_DICT['left_shoulder']][:2]],
                axis=0) 
        else:
            body.xyz_ref = None
            return
        half_zone_size = int(max(body.rect_w_a / 90, 4))
        xc = int(body.xyz_ref_coords_pixel[0] + self.crop_w)
        yc = int(body.xyz_ref_coords_pixel[1])
        roi_left = max(0, xc - half_zone_size)
        roi_right = min(self.img_w-1, xc + half_zone_size)
        roi_top = max(0, yc - half_zone_size)
        roi_bottom = min(self.img_h-1, yc + half_zone_size)
        roi_topleft = dai.Point2f(roi_left, roi_top)
        roi_bottomright = dai.Point2f(roi_right, roi_bottom)
        conf_data = dai.SpatialLocationCalculatorConfigData()
        conf_data.depthThresholds.lowerThreshold = 100
        conf_data.depthThresholds.upperThreshold = 10000
        conf_data.roi = dai.Rect(roi_topleft, roi_bottomright)
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.setROIs([conf_data])
        self.q_spatial_config.send(cfg)
        spatial_data = self.q_spatial_data.get().getSpatialLocations()
        sd = spatial_data[0]
        body.xyz_zone =  [
            int(sd.config.roi.topLeft().x) - self.crop_w,
            int(sd.config.roi.topLeft().y),
            int(sd.config.roi.bottomRight().x) - self.crop_w,
            int(sd.config.roi.bottomRight().y)
            ]
        body.xyz = np.array([
            sd.spatialCoordinates.x,
            sd.spatialCoordinates.y,
            sd.spatialCoordinates.z
            ])
        if self.smoothing:
            body.xyz = self.filter_xyz.apply(body.xyz)
    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("Identity_1"), dtype=np.float16) 
        bboxes = np.array(inference.getLayerFp16("Identity"), dtype=np.float16).reshape((self.nb_anchors,12)) 
        bodies = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors, best_only=True)
        if bodies:
            body = bodies[0]
        else:
            return None
        mpu.detections_to_rect(body)
        mpu.rect_transformation(body, self.frame_size, self.frame_size, self.rect_transf_scale)
        return body
    def lm_postprocess(self, body, inference):
        body.lm_score = inference.getLayerFp16("Identity_1")[0]
        if body.lm_score > self.lm_score_thresh:  
            lm_raw = np.array(inference.getLayerFp16("Identity")).reshape(-1,5)
            lm_raw[:,:3] /= self.lm_input_length
            body.visibility = 1 / (1 + np.exp(-lm_raw[:,3]))
            body.presence = 1 / (1 + np.exp(-lm_raw[:,4]))
            body.norm_landmarks = lm_raw[:,:3]
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in body.rect_points[1:]], dtype=np.float32) 
            mat = cv2.getAffineTransform(src, dst)
            lm_xy = np.expand_dims(body.norm_landmarks[:self.nb_kps+2,:2], axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat))  
            lm_z = body.norm_landmarks[:self.nb_kps+2,2:3] * body.rect_w_a / 4
            lm_xyz = np.hstack((lm_xy, lm_z))
            body.landmarks_world = np.array(inference.getLayerFp16("Identity_4")).reshape(-1,3)[:self.nb_kps]
            sin_rot = sin(body.rotation)
            cos_rot = cos(body.rotation)
            rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
            body.landmarks_world[:,:2] = np.dot(body.landmarks_world[:,:2], rot_m)
            if self.smoothing:
                timestamp = now()
                object_scale = body.rect_w_a
                lm_xyz[:self.nb_kps] = self.filter_landmarks.apply(lm_xyz[:self.nb_kps], timestamp, object_scale)
                lm_xyz[self.nb_kps:] = self.filter_landmarks_aux.apply(lm_xyz[self.nb_kps:], timestamp, object_scale)
                body.landmarks_world = self.filter_landmarks_world.apply(body.landmarks_world, timestamp)
            body.landmarks = lm_xyz.astype(np.int32)
            self.body_from_landmarks = mpu.Body(pd_kps=body.landmarks[self.nb_kps:self.nb_kps+2,:2]/self.frame_size)
            if self.pad_h > 0:
                body.landmarks[:,1] -= self.pad_h
                for i in range(len(body.rect_points)):
                    body.rect_points[i][1] -= self.pad_h
            if self.pad_w > 0:
                body.landmarks[:,0] -= self.pad_w
                for i in range(len(body.rect_points)):
                    body.rect_points[i][0] -= self.pad_w
    def next_frame(self):
        self.fps.update()
        if self.input_type == "rgb":
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()
            if self.pad_h:
                square_frame = cv2.copyMakeBorder(video_frame, self.pad_h, self.pad_h, self.pad_w, self.pad_w, cv2.BORDER_CONSTANT)
            else:
                square_frame = video_frame
        if self.force_detection or not self.use_previous_landmarks:
            if self.input_type == "rgb":
                self.q_pre_pd_manip_cfg.send(self.cfg_pre_pd)
            inference = self.q_pd_out.get()
            if self.input_type != "rgb" and (self.force_detection or not self.use_previous_landmarks): 
                pd_rtrip_time = now() - pd_rtrip_time
                self.glob_pd_rtrip_time += pd_rtrip_time
            body = self.pd_postprocess(inference)
            self.nb_pd_inferences += 1
        else:
            body = self.body_from_landmarks
            mpu.detections_to_rect(body) 
            mpu.rect_transformation(body, self.frame_size, self.frame_size, self.rect_transf_scale)
        if body:
            frame_nn = mpu.warp_rect_img(body.rect_points, square_frame, self.lm_input_length, self.lm_input_length)
            frame_nn = frame_nn / 255.
            nn_data = dai.NNData()   
            nn_data.setLayer("input_1", to_planar(frame_nn, (self.lm_input_length, self.lm_input_length)))
            lm_rtrip_time = now()
            self.q_lm_in.send(nn_data)
            inference = self.q_lm_out.get()
            lm_rtrip_time = now() - lm_rtrip_time
            self.glob_lm_rtrip_time += lm_rtrip_time
            self.nb_lm_inferences += 1
            self.lm_postprocess(body, inference)
            if body.lm_score < self.lm_score_thresh:
                body = None
                self.use_previous_landmarks = False
                if self.smoothing: 
                    self.filter_landmarks.reset()
                    self.filter_landmarks_aux.reset()
                    self.filter_landmarks_world.reset()
            else:
                self.use_previous_landmarks = True
                if self.xyz:
                    self.query_body_xyz(body)
        else:
            self.use_previous_landmarks = False
            if self.smoothing: 
                self.filter_landmarks.reset()
                self.filter_landmarks_aux.reset()
                self.filter_landmarks_world.reset()
                if self.xyz: self.filter_xyz.reset()
        return video_frame, body
    def exit(self):
        self.device.close()
