import React, { useState, useEffect, useRef } from 'react';
import { Camera, RotateCcw, Play, Pause, Trophy, Brain, Save } from 'lucide-react';

const BadmintonScorer = () => {
  const [leftScore, setLeftScore] = useState(0);
  const [rightScore, setRightScore] = useState(0);
  const [gameMode, setGameMode] = useState(21); // 11 or 21 points
  const [isPlaying, setIsPlaying] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [gestureDetected, setGestureDetected] = useState('');
  const [winner, setWinner] = useState(null);
  const [trainingMode, setTrainingMode] = useState(false);
  const [trainingSamples, setTrainingSamples] = useState({ left: [], right: [], none: [] });
  const [model, setModel] = useState(null);
  const [useAI, setUseAI] = useState(false);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handsRef = useRef(null);
  const cameraRef = useRef(null);
  const lastGestureTimeRef = useRef(0);
  const gestureDebounceRef = useRef(300); // 防止重複計分的延遲時間（毫秒）

  // 初始化 MediaPipe Hands 和 TensorFlow.js
  useEffect(() => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js';
    script.async = true;
    document.body.appendChild(script);

    const cameraScript = document.createElement('script');
    cameraScript.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js';
    cameraScript.async = true;
    document.body.appendChild(cameraScript);

    const tfScript = document.createElement('script');
    tfScript.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.11.0/dist/tf.min.js';
    tfScript.async = true;
    document.body.appendChild(tfScript);

    return () => {
      document.body.removeChild(script);
      document.body.removeChild(cameraScript);
      document.body.removeChild(tfScript);
    };
  }, []);

  // 將手部關鍵點轉換為特徵向量
  const landmarksToFeatures = (landmarks) => {
    if (!landmarks || landmarks.length === 0) return null;
    
    const hand = landmarks[0];
    // 提取所有 21 個關鍵點的 x, y 座標（共 42 個特徵）
    const features = [];
    for (let i = 0; i < hand.length; i++) {
      features.push(hand[i].x, hand[i].y);
    }
    return features;
  };

  // AI 手勢預測
  const predictGestureAI = async (landmarks) => {
    if (!model || !landmarks || landmarks.length === 0) return null;
    
    const features = landmarksToFeatures(landmarks);
    if (!features) return null;

    try {
      const input = window.tf.tensor2d([features]);
      const prediction = model.predict(input);
      const probabilities = await prediction.data();
      input.dispose();
      prediction.dispose();

      // 找出最高機率的類別
      const maxProb = Math.max(...probabilities);
      const maxIndex = probabilities.indexOf(maxProb);

      // 只有當信心度 > 0.7 時才返回結果
      if (maxProb > 0.7) {
        return ['left', 'right', 'none'][maxIndex];
      }
    } catch (error) {
      console.error('AI 預測錯誤:', error);
    }
    
    return null;
  };

  // 訓練 AI 模型
  const trainModel = async () => {
    if (!window.tf) {
      alert('TensorFlow.js 尚未載入，請稍後再試');
      return;
    }

    const totalSamples = trainingSamples.left.length + 
                        trainingSamples.right.length + 
                        trainingSamples.none.length;
    
    if (totalSamples < 30) {
      alert('訓練樣本太少！建議每個手勢至少收集 10 個樣本');
      return;
    }

    try {
      // 準備訓練資料
      const xs = [];
      const ys = [];

      trainingSamples.left.forEach(sample => {
        xs.push(sample);
        ys.push([1, 0, 0]); // 左邊得分
      });

      trainingSamples.right.forEach(sample => {
        xs.push(sample);
        ys.push([0, 1, 0]); // 右邊得分
      });

      trainingSamples.none.forEach(sample => {
        xs.push(sample);
        ys.push([0, 0, 1]); // 無動作
      });

      const xsTensor = window.tf.tensor2d(xs);
      const ysTensor = window.tf.tensor2d(ys);

      // 建立神經網路模型
      const newModel = window.tf.sequential({
        layers: [
          window.tf.layers.dense({ inputShape: [42], units: 64, activation: 'relu' }),
          window.tf.layers.dropout({ rate: 0.2 }),
          window.tf.layers.dense({ units: 32, activation: 'relu' }),
          window.tf.layers.dense({ units: 3, activation: 'softmax' })
        ]
      });

      newModel.compile({
        optimizer: window.tf.train.adam(0.001),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
      });

      setGestureDetected('訓練中...');

      // 訓練模型
      await newModel.fit(xsTensor, ysTensor, {
        epochs: 50,
        batchSize: 8,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
            if (epoch % 10 === 0) {
              setGestureDetected(`訓練進度: ${epoch}/50 - 準確度: ${(logs.acc * 100).toFixed(1)}%`);
            }
          }
        }
      });

      xsTensor.dispose();
      ysTensor.dispose();

      setModel(newModel);
      setUseAI(true);
      setGestureDetected('訓練完成！AI 模式已啟動');
      setTimeout(() => setGestureDetected(''), 2000);

      alert(`模型訓練完成！\n訓練樣本: ${totalSamples} 個\n現在可以使用 AI 辨識手勢了`);
    } catch (error) {
      console.error('訓練錯誤:', error);
      alert('訓練失敗：' + error.message);
    }
  };

  // 收集訓練樣本
  const collectSample = (landmarks, label) => {
    const features = landmarksToFeatures(landmarks);
    if (!features) return;

    setTrainingSamples(prev => ({
      ...prev,
      [label]: [...prev[label], features]
    }));

    setGestureDetected(`已收集 ${label} 樣本 x${trainingSamples[label].length + 1}`);
    setTimeout(() => setGestureDetected(''), 800);
  };
  const detectGesture = (landmarks) => {
    if (!landmarks || landmarks.length === 0) return null;

    const hand = landmarks[0];
    
    // 取得手指尖端和基部的座標
    const thumbTip = hand[4];
    const indexTip = hand[8];
    const middleTip = hand[12];
    const ringTip = hand[16];
    const pinkyTip = hand[20];
    
    const indexBase = hand[5];
    const middleBase = hand[9];
    const ringBase = hand[13];
    const pinkyBase = hand[17];

    // 判斷手指是否伸直（tip 的 y 座標小於 base）
    const indexUp = indexTip.y < indexBase.y;
    const middleUp = middleTip.y < middleBase.y;
    const ringUp = ringTip.y < ringBase.y;
    const pinkyUp = pinkyTip.y < pinkyBase.y;

    // 計算伸直的手指數
    let fingersUp = 0;
    if (indexUp) fingersUp++;
    if (middleUp) fingersUp++;
    if (ringUp) fingersUp++;
    if (pinkyUp) fingersUp++;

    // 判斷手的位置（左側或右側）
    const handX = hand[0].x; // 手腕的 x 座標
    const isLeftSide = handX < 0.5;

    // 一根手指 = 左邊得分，兩根手指 = 右邊得分
    if (fingersUp === 1) {
      return 'left';
    } else if (fingersUp === 2) {
      return 'right';
    }

    return null;
  };

  // 處理手勢結果
  const onResults = async (results) => {
    if (!canvasRef.current || !videoRef.current) return;

    // 儲存最新的手部關鍵點供訓練模式使用
    if (handsRef.current && results.multiHandLandmarks) {
      handsRef.current.lastLandmarks = results.multiHandLandmarks;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;

    ctx.save();
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      // 繪製手部骨架
      for (const landmarks of results.multiHandLandmarks) {
        // 繪製連接線
        const connections = window.HANDS?.HAND_CONNECTIONS || [];
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        
        for (const connection of connections) {
          const start = landmarks[connection[0]];
          const end = landmarks[connection[1]];
          ctx.beginPath();
          ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
          ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
          ctx.stroke();
        }

        // 繪製關節點
        ctx.fillStyle = '#FF0000';
        for (const landmark of landmarks) {
          ctx.beginPath();
          ctx.arc(
            landmark.x * canvas.width,
            landmark.y * canvas.height,
            5,
            0,
            2 * Math.PI
          );
          ctx.fill();
        }
      }

      // 偵測手勢
      const gesture = useAI 
        ? await predictGestureAI(results.multiHandLandmarks)
        : detectGesture(results.multiHandLandmarks);
      
      const currentTime = Date.now();
      
      // 訓練模式：收集樣本
      if (trainingMode && results.multiHandLandmarks.length > 0) {
        // 不自動計分，等待用戶點擊收集按鈕
        return;
      }
      
      if (gesture && isPlaying && !winner) {
        // 防止重複計分
        if (currentTime - lastGestureTimeRef.current > gestureDebounceRef.current) {
          if (gesture === 'left') {
            setLeftScore(prev => prev + 1);
            setGestureDetected(useAI ? '🤖 AI: 左邊得分！' : '左邊得分！(1根手指)');
            lastGestureTimeRef.current = currentTime;
          } else if (gesture === 'right') {
            setRightScore(prev => prev + 1);
            setGestureDetected(useAI ? '🤖 AI: 右邊得分！' : '右邊得分！(2根手指)');
            lastGestureTimeRef.current = currentTime;
          }
          
          // 清除手勢提示
          setTimeout(() => setGestureDetected(''), 1000);
        }
      }
    }

    ctx.restore();
  };

  // 啟動相機
  const startCamera = async () => {
    if (!window.Hands || !window.Camera) {
      alert('MediaPipe 資源載入中，請稍後再試');
      return;
    }

    try {
      const hands = new window.Hands({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
        }
      });

      hands.setOptions({
        maxNumHands: 1,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
      });

      hands.onResults(onResults);
      handsRef.current = hands;
      handsRef.current.lastLandmarks = null; // 用於訓練模式

      if (videoRef.current) {
        const camera = new window.Camera(videoRef.current, {
          onFrame: async () => {
            await hands.send({ image: videoRef.current });
          },
          width: 640,
          height: 480,
          facingMode: 'user'
        });
        
        camera.start();
        cameraRef.current = camera;
        setCameraActive(true);
      }
    } catch (error) {
      console.error('相機啟動失敗:', error);
      alert('相機啟動失敗，請確認已授權相機權限');
    }
  };

  // 停止相機
  const stopCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
    }
    setCameraActive(false);
  };

  // 檢查獲勝條件
  useEffect(() => {
    if (!isPlaying) return;

    const winScore = gameMode;
    const scoreDiff = Math.abs(leftScore - rightScore);

    if (leftScore >= winScore && scoreDiff >= 2) {
      setWinner('left');
      setIsPlaying(false);
    } else if (rightScore >= winScore && scoreDiff >= 2) {
      setWinner('right');
      setIsPlaying(false);
    }
  }, [leftScore, rightScore, gameMode, isPlaying]);

  // 重置遊戲
  const resetGame = () => {
    setLeftScore(0);
    setRightScore(0);
    setWinner(null);
    setIsPlaying(false);
    setGestureDetected('');
  };

  // 開始/暫停比賽
  const togglePlay = () => {
    if (!cameraActive) {
      alert('請先啟動相機');
      return;
    }
    setIsPlaying(!isPlaying);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 via-blue-800 to-purple-900 p-4">
      <div className="max-w-6xl mx-auto">
        {/* 標題 */}
        <div className="text-center mb-6">
          <h1 className="text-4xl font-bold text-white mb-2">🏸 羽球手勢計分系統</h1>
          <p className="text-blue-200">使用手勢控制：1根手指 = 左邊得分 / 2根手指 = 右邊得分</p>
        </div>

        {/* 相機視窗 */}
        <div className="bg-black rounded-lg overflow-hidden mb-6 relative">
          <video
            ref={videoRef}
            className="hidden"
            playsInline
          />
          <canvas
            ref={canvasRef}
            className="w-full h-auto"
            style={{ maxHeight: '400px' }}
          />
          
          {!cameraActive && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
              <button
                onClick={startCamera}
                className="bg-blue-500 hover:bg-blue-600 text-white px-8 py-4 rounded-lg font-bold text-xl flex items-center gap-2"
              >
                <Camera size={24} />
                啟動相機
              </button>
            </div>
          )}

          {gestureDetected && (
            <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-green-500 text-white px-6 py-3 rounded-lg font-bold text-xl animate-pulse">
              {gestureDetected}
            </div>
          )}
        </div>

        {/* 計分板 */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          {/* 左邊分數 */}
          <div className="bg-gradient-to-br from-red-500 to-red-700 rounded-lg p-6 text-center relative">
            <h2 className="text-white text-2xl font-bold mb-2">左邊</h2>
            <div className="text-8xl font-bold text-white">{leftScore}</div>
            {winner === 'left' && (
              <div className="absolute top-4 right-4">
                <Trophy size={48} className="text-yellow-300" />
              </div>
            )}
          </div>

          {/* 右邊分數 */}
          <div className="bg-gradient-to-br from-blue-500 to-blue-700 rounded-lg p-6 text-center relative">
            <h2 className="text-white text-2xl font-bold mb-2">右邊</h2>
            <div className="text-8xl font-bold text-white">{rightScore}</div>
            {winner === 'right' && (
              <div className="absolute top-4 right-4">
                <Trophy size={48} className="text-yellow-300" />
              </div>
            )}
          </div>
        </div>

        {/* 勝利訊息 */}
        {winner && (
          <div className="bg-yellow-400 text-gray-900 text-center py-4 rounded-lg mb-6 font-bold text-2xl">
            🎉 {winner === 'left' ? '左邊' : '右邊'}獲勝！🎉
          </div>
        )}

        {/* 控制按鈕 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
          <button
            onClick={togglePlay}
            disabled={!cameraActive || trainingMode}
            className={`py-4 rounded-lg font-bold text-white flex items-center justify-center gap-2 ${
              cameraActive && !trainingMode
                ? 'bg-green-500 hover:bg-green-600'
                : 'bg-gray-500 cursor-not-allowed'
            }`}
          >
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            {isPlaying ? '暫停' : '開始'}
          </button>

          <button
            onClick={resetGame}
            className="bg-orange-500 hover:bg-orange-600 text-white py-4 rounded-lg font-bold flex items-center justify-center gap-2"
          >
            <RotateCcw size={20} />
            重置
          </button>

          <button
            onClick={() => setGameMode(11)}
            disabled={trainingMode}
            className={`py-4 rounded-lg font-bold ${
              gameMode === 11
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-300'
            } ${trainingMode ? 'cursor-not-allowed opacity-50' : ''}`}
          >
            11分制
          </button>

          <button
            onClick={() => setGameMode(21)}
            disabled={trainingMode}
            className={`py-4 rounded-lg font-bold ${
              gameMode === 21
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-300'
            } ${trainingMode ? 'cursor-not-allowed opacity-50' : ''}`}
          >
            21分制
          </button>
        </div>

        {/* AI 訓練區 */}
        <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 mb-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-white flex items-center gap-2">
              <Brain size={24} />
              AI 手勢學習
            </h3>
            <button
              onClick={() => {
                setTrainingMode(!trainingMode);
                if (!trainingMode) {
                  setIsPlaying(false);
                }
              }}
              className={`px-4 py-2 rounded-lg font-bold ${
                trainingMode
                  ? 'bg-red-500 hover:bg-red-600'
                  : 'bg-blue-500 hover:bg-blue-600'
              } text-white`}
            >
              {trainingMode ? '退出訓練' : '進入訓練模式'}
            </button>
          </div>

          {trainingMode ? (
            <div className="space-y-4">
              <p className="text-white text-sm mb-4">
                對著鏡頭擺出手勢，然後點擊對應的按鈕收集樣本。建議每個手勢收集 10-20 個不同角度的樣本。
              </p>
              
              <div className="grid grid-cols-3 gap-3">
                <button
                  onClick={() => collectSample(handsRef.current?.lastLandmarks, 'left')}
                  disabled={!cameraActive}
                  className="bg-red-500 hover:bg-red-600 disabled:bg-gray-500 text-white py-3 rounded-lg font-bold"
                >
                  <div>收集「左邊」</div>
                  <div className="text-sm">({trainingSamples.left.length} 個)</div>
                </button>

                <button
                  onClick={() => collectSample(handsRef.current?.lastLandmarks, 'right')}
                  disabled={!cameraActive}
                  className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-500 text-white py-3 rounded-lg font-bold"
                >
                  <div>收集「右邊」</div>
                  <div className="text-sm">({trainingSamples.right.length} 個)</div>
                </button>

                <button
                  onClick={() => collectSample(handsRef.current?.lastLandmarks, 'none')}
                  disabled={!cameraActive}
                  className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-500 text-white py-3 rounded-lg font-bold"
                >
                  <div>收集「無動作」</div>
                  <div className="text-sm">({trainingSamples.none.length} 個)</div>
                </button>
              </div>

              <button
                onClick={trainModel}
                disabled={trainingSamples.left.length < 5 || trainingSamples.right.length < 5}
                className="w-full bg-green-500 hover:bg-green-600 disabled:bg-gray-500 disabled:cursor-not-allowed text-white py-4 rounded-lg font-bold flex items-center justify-center gap-2"
              >
                <Save size={20} />
                開始訓練 AI 模型
              </button>

              <div className="text-white text-sm space-y-1">
                <div>📊 訓練狀態：</div>
                <div>• 左邊樣本：{trainingSamples.left.length} 個</div>
                <div>• 右邊樣本：{trainingSamples.right.length} 個</div>
                <div>• 無動作樣本：{trainingSamples.none.length} 個</div>
                <div>• AI 模型：{model ? '✅ 已訓練' : '❌ 未訓練'}</div>
                <div>• 使用模式：{useAI ? '🤖 AI 模式' : '📏 規則模式'}</div>
              </div>
            </div>
          ) : (
            <div className="text-white space-y-2">
              <p>目前使用：{useAI ? '🤖 AI 辨識模式' : '📏 規則辨識模式'}</p>
              <p className="text-sm text-gray-300">
                {useAI 
                  ? '正在使用你訓練的 AI 模型進行手勢辨識'
                  : '使用預設的手指計數規則：1根手指=左邊，2根手指=右邊'
                }
              </p>
              {model && (
                <button
                  onClick={() => setUseAI(!useAI)}
                  className="mt-2 px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg font-bold"
                >
                  切換到 {useAI ? '規則模式' : 'AI 模式'}
                </button>
              )}
            </div>
          )}
        </div>

        {/* 控制按鈕 */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <button
            onClick={togglePlay}
            disabled={!cameraActive}
            className={`py-4 rounded-lg font-bold text-white flex items-center justify-center gap-2 ${
              cameraActive
                ? 'bg-green-500 hover:bg-green-600'
                : 'bg-gray-500 cursor-not-allowed'
            }`}
          >
            {isPlaying ? <Pause size={20} /> : <Play size={20} />}
            {isPlaying ? '暫停' : '開始'}
          </button>

          <button
            onClick={resetGame}
            className="bg-orange-500 hover:bg-orange-600 text-white py-4 rounded-lg font-bold flex items-center justify-center gap-2"
          >
            <RotateCcw size={20} />
            重置
          </button>

          <button
            onClick={() => setGameMode(11)}
            disabled={trainingMode}
            className={`py-4 rounded-lg font-bold ${
              gameMode === 11
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-300'
            } ${trainingMode ? 'cursor-not-allowed opacity-50' : ''}`}
          >
            11分制
          </button>

          <button
            onClick={() => setGameMode(21)}
            disabled={trainingMode}
            className={`py-4 rounded-lg font-bold ${
              gameMode === 21
                ? 'bg-purple-600 text-white'
                : 'bg-gray-700 text-gray-300'
            } ${trainingMode ? 'cursor-not-allowed opacity-50' : ''}`}
          >
            21分制
          </button>
        </div>

        {/* 使用說明 */}
        <div className="mt-4 bg-white/10 backdrop-blur-sm rounded-lg p-6 text-white">
          <h3 className="text-xl font-bold mb-3">📋 使用說明</h3>
          <div className="space-y-4">
            <div>
              <h4 className="font-bold mb-2">🎮 基本操作</h4>
              <ul className="space-y-1 text-sm">
                <li>• <strong>啟動相機</strong>：點擊「啟動相機」按鈕</li>
                <li>• <strong>開始比賽</strong>：點擊「開始」按鈕</li>
                <li>• <strong>規則模式</strong>：1根手指=左邊得分，2根手指=右邊得分</li>
                <li>• <strong>獲勝條件</strong>：先達到設定分數且領先2分以上</li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-bold mb-2">🤖 AI 訓練模式</h4>
              <ul className="space-y-1 text-sm">
                <li>• <strong>進入訓練</strong>：點擊「進入訓練模式」</li>
                <li>• <strong>收集樣本</strong>：擺出手勢後點擊對應按鈕，建議每個手勢收集 10-20 個樣本</li>
                <li>• <strong>訓練模型</strong>：收集足夠樣本後點擊「開始訓練 AI 模型」</li>
                <li>• <strong>使用 AI</strong>：訓練完成後會自動切換到 AI 模式</li>
                <li>• <strong>自訂手勢</strong>：可以訓練任何你喜歡的手勢！</li>
              </ul>
            </div>

            <div className="bg-yellow-500/20 border border-yellow-500 rounded p-3 text-sm">
              <strong>💡 提示：</strong>AI 模式可以讓你使用任何自訂手勢，不限於手指數量！
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BadmintonScorer;