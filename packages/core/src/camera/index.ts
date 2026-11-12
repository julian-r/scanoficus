/**
 * Browser camera plumbing for live scanning: stream acquisition, frame loop, downscaled frame
 * extraction, still capture, torch and page-visibility handling. See docs/platform-notes.md for
 * the platform behavior these choices are based on.
 *
 * @module
 */
export {
  CameraError,
  type CameraErrorCode,
  type CameraFeatures,
  type CameraOptions,
  type CameraSession,
  type CameraStartStage,
  listCameras,
  openCamera,
  type StillCapture,
  toCameraError,
} from './camera.ts';
export {
  FrameGrabber,
  type FrameLoopOptions,
  type GrabSource,
  startFrameLoop,
  watchPageVisibility,
} from './frames.ts';
export { requestDeviceMotionPermission } from './motion.ts';
