"use client";

import { Button } from "@/components/ui/button";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";

const PoseDetection: React.FC = () => {
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 });
  const [isLoading, setLoading] = useState(true);
  const [currentPosture, setCurrentPosture] =
    useState<poseDetection.Pose | null>(null);
  const [idealPosture, setIdealPosture] = useState<poseDetection.Pose | null>();
  const [postureStatus, setPostureStatus] = useState("");
  const videoRef = useRef<HTMLVideoElement | null>(null);

  function onSaveIdealPosture() {
    setIdealPosture(currentPosture);
  }

  useEffect(() => {
    const setupCamera = async () => {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error(
          "Browser API navigator.mediaDevices.getUserMedia not available"
        );
      }
      const video = videoRef.current;
      if (!video) return;
      video.width = dimensions.width;
      video.height = dimensions.height;
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

    const evaluatePosture = (pose: poseDetection.Pose | null) => {
      if (!pose) {
        setPostureStatus("No pose detected");
        return;
      }

      // Assuming keypoints are available
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

      const maxLevelDifference = 10;
      const maxHorizontalDistance = 30;

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

    const detectPoseInRealTime = async (
      video: HTMLVideoElement,
      detector: poseDetection.PoseDetector
    ) => {
      const poseDetectionFrame = async () => {
        const [pose] = await detector.estimatePoses(video);

        setCurrentPosture(pose);

        evaluatePosture(pose);

        requestAnimationFrame(poseDetectionFrame);

        setLoading(false);
      };

      poseDetectionFrame();
    };

    const init = async () => {
      await tf.setBackend("webgl");
      await tf.ready();

      const video = await setupCamera();

      if (video) {
        video.play();

        const detector = await loadPoseModel();

        detectPoseInRealTime(video, detector);
      }
    };

    init();
  }, [dimensions.height, dimensions.width]);

  useLayoutEffect(() => {
    const updateDimensions = () => {
      const width = window.innerWidth;

      const targetWidth = width > 640 ? 640 : width;
      const targetHeight = (targetWidth * 480) / 640;

      setDimensions({ width: targetWidth, height: targetHeight });
    };

    updateDimensions();

    window.addEventListener("resize", updateDimensions);

    return () => window.removeEventListener("resize", updateDimensions);
  }, []);

  return (
    <>
      {isLoading && <div>Loading...</div>}
      {postureStatus}
      <div className="relative">
        <video ref={videoRef} className="scale-x-[-1]" />
      </div>
      <Button onClick={onSaveIdealPosture}>Save</Button>
      {idealPosture &&
        idealPosture.keypoints.map(({ x, y, name }) => {
          if (
            name &&
            ["nose", "left_shoulder", "right_shoulder"].includes(name)
          )
            return (
              <div key={name}>
                {x} {y} {name}
              </div>
            );
        })}
    </>
  );
};

export default PoseDetection;
