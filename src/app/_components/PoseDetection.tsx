"use client";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
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
import Icon from "./../../../public/icon.webp";

const PoseDetection = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [loading, setLoading] = useState(true);
  const [dimensions, setDimensions] = useState({ width: 640, height: 480 });
  const [video, setVideo] = useState<HTMLVideoElement | null>(null);
  const [detector, setDetector] = useState<PoseDetector | null>(null);
  const [idealPosture, setIdealPosture] = useState<Pose | null>(null);
  const [isBadPosture, setIsBadPosture] = useState(false);
  const [badPostureTimeoutId, setBadPostureTimeoutId] =
    useState<NodeJS.Timeout | null>(null);
  const [isStartSession, setIsStartSession] = useState(false);
  const [screenshots, setScreenshots] = useState<string[]>([]);
  const [duration, setDuration] = useState(5);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  const onSave = async () => {
    if (!detector || !video) return;
    const [pose] = await detector.estimatePoses(video);
    setIdealPosture(pose);
  };

  // init notification permission
  useEffect(() => {
    if (
      Notification.permission !== "granted" &&
      Notification.permission !== "denied"
    ) {
      Notification.requestPermission().then((permission) => {
        if (permission === "granted") {
          console.log("Notification permission granted.");
        }
      });
    }
  }, []);

  // notification
  useEffect(() => {
    if (!isStartSession) return;

    if (isBadPosture && !badPostureTimeoutId) {
      const timeoutId = setTimeout(() => {
        if ("Notification" in window && Notification.permission === "granted") {
          new Notification("PoseMax: Adjust Your Posture", {
            body: "You've been in a bad posture for a while. Time to straighten up!",
            icon: Icon.src,
          });

          if (videoRef.current) {
            const canvas = document.createElement("canvas");
            canvas.width = 240;
            canvas.height = 240;

            const ctx = canvas.getContext("2d");
            if (ctx) {
              ctx.drawImage(
                videoRef.current,
                0,
                0,
                canvas.width,
                canvas.height
              );
            }

            const dataURL = canvas.toDataURL("image/png");

            setScreenshots((prev) => [...prev, dataURL]);
          }

          setBadPostureTimeoutId(null);
        }
      }, duration * 1000);

      setBadPostureTimeoutId(timeoutId);
    } else if (!isBadPosture && badPostureTimeoutId) {
      clearTimeout(badPostureTimeoutId);
      setBadPostureTimeoutId(null);
    }

    return () => {
      if (badPostureTimeoutId) {
        clearTimeout(badPostureTimeoutId);
      }
    };
  }, [isBadPosture, badPostureTimeoutId, isStartSession, duration]);

  // init camera and load model
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

  // run to evalute posture
  useEffect(() => {
    if (!video || !detector || !video.width || !video.height) return;

    const evaluatePosture = (pose: Pose | null) => {
      if (!pose || !idealPosture) {
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

      const HORIZONTAL_DIFFERENCE_THRESHOLD = 60;
      const SHOULDER_LEVEL_DIFFERENCE_THRESHOLD = 30;
      const SHOULDER_VERTICAL_THRESHOLD = 30;

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
        setIsBadPosture(false);
      } else {
        setIsBadPosture(true);
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
  }, [detector, idealPosture, video]);

  // resize screen
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
      <div className="flex justify-center">
        {loading && <Spinner className="w-60 h-60 mt-32" />}
      </div>
      <div
        className={cn(
          "flex flex-col items-center gap-6 max-w-[40rem]",
          loading && "hidden"
        )}
      >
        <p className={cn("leading-7 mr-auto", idealPosture && "hidden")}>
          Please set your ideal posture now. Position yourself in the most
          comfortable, upright posture you believe is ideal for you. This will
          help us compare and improve your posture over time.
        </p>

        <div className="relative">
          <div
            className={cn(
              "hidden opacity-0",
              idealPosture && "block",
              isStartSession && "opacity-100"
            )}
          >
            {isBadPosture ? (
              <p className="text-center font-extrabold text-red-700">
                BAD POSTURE
              </p>
            ) : (
              <p className="text-center font-extrabold text-green-700">
                GOOD POSTURE
              </p>
            )}
          </div>
          <video
            ref={videoRef}
            className={cn(
              "scale-x-[-1] rounded-2xl border-4 border-black",
              isStartSession &&
                (isBadPosture ? "border-red-600" : "border-green-600")
            )}
            width={dimensions.width}
            height={dimensions.height}
          />
        </div>
        <div>
          {idealPosture ? (
            <div className="flex gap-4">
              {isStartSession ? (
                <Button onClick={() => setIsStartSession(false)}>
                  End Session
                </Button>
              ) : (
                <div className="flex w-full sm:items-end flex-col sm:flex-row gap-4">
                  <div>
                    <Label htmlFor="duration" className="text-xs">
                      Set Notification Delay (in seconds)
                    </Label>
                    <Input
                      className="bg-white"
                      type="number"
                      id="duration"
                      step={1}
                      value={duration}
                      min={1}
                      onChange={(e) =>
                        setDuration(parseInt(e.currentTarget.value))
                      }
                    />
                  </div>
                  <div className="space-x-4">
                    <Button
                      onClick={() => {
                        setScreenshots([]);
                        setIsStartSession(true);
                      }}
                    >
                      Start Session
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => {
                        setScreenshots([]);
                        setIdealPosture(null);
                      }}
                    >
                      Reset Ideal Posture
                    </Button>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <Button className="w-fit" onClick={onSave}>
              Save Ideal Posture
            </Button>
          )}
        </div>
      </div>
      {screenshots.length > 0 && (
        <div className="mt-16 space-y-6">
          <div>
            <h3 className="scroll-m-20 text-center text-2xl font-semibold tracking-tight">
              Recap Your Bad Posture
            </h3>
          </div>
          <div className="flex items-start overflow-x-scroll  max-w-7xl shadow-inner border-2 border-black rounded-xl">
            {screenshots.map((screenshot, index) => (
              <Dialog key={`Screenshot ${index + 1}`}>
                <DialogTrigger
                  className={cn(
                    "shrink-0 border-r border-black hover:opacity-80 transition-opacity"
                  )}
                >
                  <div>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={screenshot}
                      className="scale-x-[-1]"
                      alt={`Screenshot ${index + 1}`}
                      onClick={() => setSelectedImage(screenshot)}
                    />
                  </div>
                </DialogTrigger>
                <DialogContent>
                  <div
                    className="selected-image-container"
                    style={{ textAlign: "center", marginTop: "20px" }}
                  >
                    {selectedImage && (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={selectedImage}
                        alt="Selected Screenshot"
                        className="w-full h-full object-cover rounded-2xl scale-x-[-1]"
                      />
                    )}
                  </div>
                </DialogContent>
              </Dialog>
            ))}
          </div>
        </div>
      )}
    </>
  );
};

export default PoseDetection;
