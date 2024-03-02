"use client";

import { cn } from "@/lib/utils";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import debounce from "lodash/debounce";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

const PoseDetection = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [loading, setLoading] = useState(true);
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 });
  const [screenRatio, setScreenRatio] = useState<number>(
    window.innerWidth / window.innerHeight
  );
  const [postureStatus, setPostureStatus] = useState("");
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [detector, setDetector] = useState<poseDetection.PoseDetector | null>(
    null
  );

  useEffect(() => {
    if (!video || !detector || video.width <= 0 || video.height <= 0) return;

    const evaluatePosture = (pose: poseDetection.Pose | null) => {
      if (!pose) {
        setPostureStatus("No pose detected");
        return;
      }

      const nose = pose.keypoints.find((k) => k.name === "nose");
      const leftShoulder = pose.keypoints.find(
        (k) => k.name === "left_shoulder"
      );
      const rightShoulder = pose.keypoints.find(
        (k) => k.name === "right_shoulder"
      );

      if (!nose || !leftShoulder || !rightShoulder) {
        setPostureStatus("Essential keypoints missing");
        return;
      }

      const shoulderMidpointX = (leftShoulder.x + rightShoulder.x) / 2;
      const horizontalDistance = Math.abs(nose.x - shoulderMidpointX);
      const shoulderLevelDifference = Math.abs(
        leftShoulder.y - rightShoulder.y
      );

      const maxLevelDifference = 10 * screenRatio;
      const maxHorizontalDistance = 30 * screenRatio;

      const isLeaningForward = horizontalDistance > maxHorizontalDistance;
      const isTilting = shoulderLevelDifference > maxLevelDifference;

      if (isLeaningForward && isTilting) {
        setPostureStatus("Leaning forward and tilting");
      } else if (isLeaningForward) {
        setPostureStatus("Leaning forward");
      } else if (isTilting) {
        setPostureStatus("Tilting");
      } else {
        setPostureStatus("Good posture");
      }
    };

    const detectPoseInRealTime = async () => {
      const poseDetectionFrame = async () => {
        const [pose] = await detector.estimatePoses(video);
        evaluatePosture(pose);
        requestAnimationFrame(poseDetectionFrame);
      };

      poseDetectionFrame();
    };

    const run = async () => {
      video.play();
      detectPoseInRealTime();
      setLoading(false);
    };

    run();
  }, [detector, screenRatio, video]);

  useEffect(() => {
    const setupCamera = async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
          "Browser API navigator.mediaDevices.getUserMedia not available"
        );
      }
      const video = videoRef.current;

      if (!video) return;

      video.srcObject = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: "user",
          width: video.width,
          height: video.height,
        },
      });

      return new Promise<HTMLVideoElement>((resolve) => {
        video.onloadedmetadata = () => {
          resolve(video);
        };
      });
    };

    const loadPoseModel = async () => {
      const detectorConfig = {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
      };

      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        detectorConfig
      );

      return detector;
    };

    const init = async () => {
      await tf.setBackend("webgl");
      await tf.ready();

      const video = await setupCamera();
      const detector = await loadPoseModel();

      if (video) {
        setVideo(video);
      }

      if (detector) {
        setDetector(detector);
      }
    };

    init();
  }, []);

  useLayoutEffect(() => {
    const updateDimensions = debounce(() => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      const targetWidth = width > 640 ? 640 : width;
      const targetHeight = (targetWidth * 480) / 640;
      const newScreenRatio = width / height;
      setDimensions({ width: targetWidth, height: targetHeight });
      setScreenRatio(newScreenRatio);
    }, 100);
    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  return (
    <>
      {loading && <div>Loading...</div>}
      <div className={cn(loading && "hidden")}>
        <div className="text-center">Status: {postureStatus}</div>
        <video
          ref={videoRef}
          className="scale-x-[-1]"
          width={dimensions.width}
          height={dimensions.height}
        />
      </div>
    </>
  );
};

export default PoseDetection;
