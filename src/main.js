import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import "@tensorflow/tfjs-backend-webgl";

const appDom = document.getElementById("app");

// video要素
let video = document.createElement("video");
video.autoplay = true;
video.playsInline = true;

// video画面のプレビュー
let previewCanvas = document.createElement("canvas");
let previewCtx = previewCanvas.getContext("2d");
previewCanvas.id = "previewCanvas";
appDom.appendChild(previewCanvas);

// hand detectionの情報を表示するビュー
let overlayCanvas = document.createElement("canvas");
let overlayCtx = overlayCanvas.getContext("2d");
overlayCanvas.id = "overlayCanvas";
appDom.appendChild(overlayCanvas);

// 撮影した画像を表示するimg
let captureImage = document.createElement("img");
captureImage.id = "captureImage";
appDom.appendChild(captureImage);

// 握手判定する閾値（手と手の距離）
const handshakeThreshold = 100;

// 連続撮影を防ぐための時間設定
const photoCooldown = 3000;
let lastPhotoTime = 0;

let detector;
let isRunning = false;
let animationFrameId;

// インカメラ（鏡面）
let facingMode = "user"; // user || environment

/**
 * カメラの初期設定
 */
async function setupCamera() {
  // カメラストリーム取得
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: facingMode
    }
  });
  video.srcObject = stream;
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  console.log("Video dimensions:", video.videoWidth, video.videoHeight); // デバッグ用

  appDom.appendChild(video);
  video.width = video.videoWidth;
  video.height = video.videoHeight;
  overlayCanvas.width = video.videoWidth;
  overlayCanvas.height = video.videoHeight;
  appDom.style.width = `${video.videoWidth}px`;
}

/**
 * hand detectorモデルの読み込み
 */
async function loadModel() {
  detector = await handPoseDetection.createDetector(
    handPoseDetection.SupportedModels.MediaPipeHands,
    { runtime: "tfjs", modelType: "lite", maxHands: 2 }
  );
  console.log("Model Loaded.");
}

/**
 * 手の検出（ループ）
 */
async function detectHands() {
  if (!detector || !isRunning) return;

  previewCanvas.width = video.videoWidth;
  previewCanvas.height = video.videoHeight;

  // インカメラの場合は鏡面
  if (facingMode === "user") {
    previewCtx.save();
    previewCtx.scale(-1, 1); // X軸反転
    previewCtx.drawImage(
      video,
      -previewCanvas.width,
      0,
      previewCanvas.width,
      previewCanvas.height
    );
    previewCtx.restore();
  } else {
    previewCtx.drawImage(video, 0, 0, previewCtx.width, previewCtx.height);
  }

  const hands = await detector.estimateHands(previewCanvas);

  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  // overlayCtx.drawImage(video, 0, 0, overlayCanvas.width, overlayCanvas.height);

  // keyPoint描画
  drawKeypoints(hands);

  if (hands.length === 2) {
    const c1 = getKeypointCenter(hands[0].keypoints);
    const c2 = getKeypointCenter(hands[1].keypoints);
    const dist = distance2D(c1, c2);

    console.log(`Hands Distance: ${dist.toFixed(2)}px`);

    overlayCtx.fillStyle = "black";
    overlayCtx.fillText(`Hands Distance: ${dist.toFixed(2)}px`, 10, 20);

    if (dist < handshakeThreshold) {
      console.log("Handshake detected!");
      takePhoto();
    }
  }

  animationFrameId = requestAnimationFrame(detectHands);
}

/**
 * 検出したポイントの描画
 * @param {Object} hands
 */
function drawKeypoints(hands) {
  hands.forEach((hand) => {
    hand.keypoints.forEach((point) => {
      if (!point || isNaN(point.x) || isNaN(point.y)) return; // 無効なデータを除外
      overlayCtx.beginPath();
      overlayCtx.arc(point.x, point.y, 5, 0, 2 * Math.PI);
      overlayCtx.fillStyle = "rgba(255,255,255,0.8)";
      overlayCtx.fill();
      // nameを表示
      overlayCtx.font = "8px Arial";
      overlayCtx.fillText(point.name, point.x + 8, point.y);
    });
  });
}

/**
 * 画像の撮影
 * @returns
 */
async function takePhoto() {
  // 前回の撮影から一定期間は停止
  let now = Date.now();
  if (now - lastPhotoTime < photoCooldown) return;
  lastPhotoTime = now;

  // 画像生成用のキャンバス
  let photoCanvas = document.createElement("canvas");
  photoCanvas.width = video.videoWidth;
  photoCanvas.height = video.videoHeight;
  let photoCtx = photoCanvas.getContext("2d");
  photoCtx.drawImage(video, 0, 0);

  // インカメラの場合は鏡面
  let src = "";
  if (facingMode === "user") {
    src = await mirrorFlip(photoCanvas.toDataURL("image/png"));
  } else {
    src = photoCanvas.toDataURL("image/png");
  }

  // imgのsrcに登録（表示）
  captureImage.src = src;

  // stopDetection(); // 撮影時に検出を一時停止
  // setTimeout(startDetection, 3000); // 3秒後に再開
}

/**
 * keypointの平均座標を取得
 * @param {Object} keypoints
 * @returns Object{x,y}
 */
function getKeypointCenter(keypoints) {
  let xsum = 0,
    ysum = 0;
  keypoints.forEach((kp) => {
    xsum += kp.x;
    ysum += kp.y;
  });
  return { x: xsum / keypoints.length, y: ysum / keypoints.length };
}

/**
 * 2点の距離を算出
 * @param {Object} a
 * @param {Object} b
 * @returns Number
 */
function distance2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * 検出開始
 */
function startDetection() {
  isRunning = true;
  detectHands();
}

/**
 * 検出の停止
 */
function stopDetection() {
  isRunning = false;
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
  }
}

/**
 * 画像を左右反転する
 * @param {String} base64
 * @returns
 */
function mirrorFlip(base64) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      canvas.width = img.width;
      canvas.height = img.height;

      // 反転処理
      ctx.save();
      ctx.scale(-1, 1);
      ctx.drawImage(img, -img.width, 0);
      ctx.restore();
      // base64で出力
      resolve(canvas.toDataURL());
    };
    img.onerror = (err) => reject(err);
    img.src = base64;
  });
}

/**
 * メイン処理
 */
async function main() {
  await setupCamera();
  await loadModel();
  startDetection();
}

main();
