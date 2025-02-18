import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import "@tensorflow/tfjs-backend-webgl";

let video = document.createElement("video");
video.autoplay = true;
video.playsInline = true;
video.width = 640;
video.height = 480;

document.body.appendChild(video);

let canvas = document.createElement("canvas");
let ctx = canvas.getContext("2d");
document.body.appendChild(canvas);

let lastPhotoTime = 0;
const photoCooldown = 3000; // 3秒間のクールダウン
let detector;
let isRunning = false;
let animationFrameId;

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480 }
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));

  document.body.appendChild(video);
  canvas.width = 640;
  canvas.height = 480;
  console.log("Video dimensions:", video.videoWidth, video.videoHeight); // デバッグ用
}

async function loadModel() {
  detector = await handPoseDetection.createDetector(
    handPoseDetection.SupportedModels.MediaPipeHands,
    { runtime: "tfjs" }
  );
}

function isHandshake(hand1, hand2) {
  if (!hand1 || !hand2) return false;

  let palm1 = hand1.keypoints.find((pt) => pt.name === "wrist");
  let palm2 = hand2.keypoints.find((pt) => pt.name === "wrist");

  if (!palm1 || !palm2) return false;

  let distance = Math.sqrt((palm1.x - palm2.x) ** 2 + (palm1.y - palm2.y) ** 2);

  return distance < 150; // しきい値（調整可能）
}

function drawKeypoints(hands) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  hands.forEach((hand) => {
    hand.keypoints.forEach((point) => {
      if (!point || isNaN(point.x) || isNaN(point.y)) return; // 無効なデータを除外
      ctx.beginPath();
      ctx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    });
  });
}

function takePhoto() {
  let now = Date.now();
  if (now - lastPhotoTime < photoCooldown) return; // クールダウン中なら撮影しない
  lastPhotoTime = now;

  let photoCanvas = document.createElement("canvas");
  photoCanvas.width = video.videoWidth;
  photoCanvas.height = video.videoHeight;
  let photoCtx = photoCanvas.getContext("2d");
  photoCtx.drawImage(video, 0, 0);
  let img = document.createElement("img");
  img.src = photoCanvas.toDataURL("image/png");
  document.body.appendChild(img);

  stopDetection(); // 撮影時に検出を一時停止
  setTimeout(startDetection, 3000); // 3秒後に再開
}

async function detectHands() {
  if (!detector || !isRunning) return;

  const hands = await detector.estimateHands(video);
  drawKeypoints(hands);

  if (hands.length === 2 && isHandshake(hands[0], hands[1])) {
    console.log("Handshake detected!");
    takePhoto();
  }

  animationFrameId = requestAnimationFrame(detectHands);
}

function startDetection() {
  isRunning = true;
  detectHands();
}

function stopDetection() {
  isRunning = false;
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }
}

async function main() {
  await setupCamera();
  await loadModel();
  startDetection();
}

main();
