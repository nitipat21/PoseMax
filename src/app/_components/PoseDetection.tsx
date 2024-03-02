"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  Pose,
  PoseDetector,
  SupportedModels,
  createDetector,
  movenet,
} from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import debounce from "lodash/debounce";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

const PoseDetection = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [loading, setLoading] = useState(true);
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 });
  const [postureStatus, setPostureStatus] = useState("");
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [detector, setDetector] = useState<PoseDetector | null>(null);
  const [idealPosture, setIdealPosture] = useState<Pose | null>(null);
  const [isBadPosture, setIsBadPosture] = useState(false);

  const onSave = async () => {
    if (!detector || !video) return;
    const [pose] = await detector.estimatePoses(video);
    setIdealPosture(pose);
  };

  useEffect(() => {
    if (!video || !detector || !video.width || !video.height) return;

    const evaluatePosture = (pose: Pose | null) => {
      if (!pose) {
        setPostureStatus("No pose detected");
        return;
      }

      if (!idealPosture) {
        setPostureStatus(
          "Please set your ideal posture now. Position yourself in the most comfortable, upright posture you believe is ideal for you. This will help us compare and improve your posture over time."
        );
        return;
      }

      // Extract current keypoints
      const noseCurrent = pose.keypoints.find((k) => k.name === "nose");
      const leftShoulderCurrent = pose.keypoints.find(
        (k) => k.name === "left_shoulder"
      );
      const rightShoulderCurrent = pose.keypoints.find(
        (k) => k.name === "right_shoulder"
      );

      // Extract ideal keypoints
      const noseIdeal = idealPosture.keypoints.find((k) => k.name === "nose");
      const leftShoulderIdeal = idealPosture.keypoints.find(
        (k) => k.name === "left_shoulder"
      );
      const rightShoulderIdeal = idealPosture.keypoints.find(
        (k) => k.name === "right_shoulder"
      );

      if (
        !leftShoulderCurrent ||
        !rightShoulderCurrent ||
        !noseCurrent ||
        !noseIdeal ||
        !leftShoulderIdeal ||
        !rightShoulderIdeal
      ) {
        setPostureStatus("Miss");
        return;
      }

      // Calculate current and ideal midpoints of shoulders
      const shoulderMidpointCurrentX =
        (leftShoulderCurrent.x + rightShoulderCurrent.x) / 2;
      const shoulderMidpointIdealX =
        (leftShoulderIdeal.x + rightShoulderIdeal.x) / 2;

      // Horizontal distance to midpoint comparison
      const horizontalDistanceCurrent = Math.abs(
        noseCurrent.x - shoulderMidpointCurrentX
      );
      const horizontalDistanceIdeal = Math.abs(
        noseIdeal.x - shoulderMidpointIdealX
      );

      // Shoulder level difference comparison
      const shoulderLevelDifferenceCurrent = Math.abs(
        leftShoulderCurrent.y - rightShoulderCurrent.y
      );
      const shoulderLevelDifferenceIdeal = Math.abs(
        leftShoulderIdeal.y - rightShoulderIdeal.y
      );

      // Calculate the average shoulder height difference from the ideal
      const averageShoulderHeightCurrent =
        (leftShoulderCurrent.y + rightShoulderCurrent.y) / 2;
      const averageShoulderHeightIdeal =
        (leftShoulderIdeal.y + rightShoulderIdeal.y) / 2;

      const HORIZONTAL_DIFFERENCE_THRESHOLD = 30;
      const SHOULDER_LEVEL_DIFFERENCE_THRESHOLD = 15;
      const SHOULDER_VERTICAL_THRESHOLD = 15;

      const isShoulderHigher =
        averageShoulderHeightCurrent + SHOULDER_VERTICAL_THRESHOLD <
        averageShoulderHeightIdeal;
      const isShouldersLower =
        averageShoulderHeightCurrent >
        averageShoulderHeightIdeal + SHOULDER_VERTICAL_THRESHOLD;
      const isLeaningForward =
        horizontalDistanceCurrent >
          horizontalDistanceIdeal + HORIZONTAL_DIFFERENCE_THRESHOLD ||
        isShoulderHigher;
      const isTilting =
        shoulderLevelDifferenceCurrent >
        shoulderLevelDifferenceIdeal + SHOULDER_LEVEL_DIFFERENCE_THRESHOLD;

      if (!isShouldersLower && !isLeaningForward && !isTilting) {
        setPostureStatus("Good posture. Keep it up!");
        setIsBadPosture(false);
        return; // Exit early if good posture
      }

      setIsBadPosture(true);
      let postureMessage = "Adjust your posture: ";
      if (isShouldersLower && isLeaningForward) {
        postureMessage =
          "Your shoulders are significantly lower, and you're leaning forward too much. Try to sit up straight and bring your shoulders back.";
      } else if (isShouldersLower && isTilting) {
        postureMessage =
          "Your shoulders are significantly lower, and your posture is tilted. Try to straighten your back and level your shoulders.";
      } else if (isLeaningForward) {
        postureMessage =
          "You're leaning forward too much. Try to align your head over your shoulders and sit up straight.";
      } else if (isTilting) {
        postureMessage =
          "Your posture is tilted. Make sure to keep your shoulders level.";
      } else if (isShouldersLower) {
        postureMessage =
          "Your shoulders are significantly lower than the ideal position. Try lifting them slightly and sitting up straight.";
      }

      setPostureStatus(postureMessage);
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
  }, [detector, idealPosture, video]);

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
        modelType: movenet.modelType.SINGLEPOSE_LIGHTNING,
      };

      const detector = await createDetector(
        SupportedModels.MoveNet,
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
      const targetWidth = width > 640 ? 640 : width;
      const targetHeight = (targetWidth * 480) / 640;
      setDimensions({ width: targetWidth, height: targetHeight });
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
        <Button onClick={onSave}>Save Ideal Posture</Button>
      </div>
    </>
  );
};

export default PoseDetection;
