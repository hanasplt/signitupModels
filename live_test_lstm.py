<!doctype html>
<html>
<head>
    <meta charset="utf-8">
    <title>FSL Stable Inference (ONNX)</title>
    <style>
        body { margin:0; background:#111; color:#0f0; font-family:sans-serif; display:flex; flex-direction:column; align-items:center; }
        #wrap { position:relative; width:640px; height:480px; margin-top: 20px; border: 3px solid #333; border-radius: 10px; overflow: hidden; }
        video, canvas { position:absolute; top:0; left:0; transform:scaleX(-1); }
        .ui-panel { width: 640px; background: #222; padding: 15px; border-radius: 0 0 10px 10px; border: 1px solid #333; text-align: center; }
        #pred { font-size: 2.8em; font-weight: bold; color: #0f0; text-shadow: 0 0 10px #0f0; margin: 5px 0; }
        #conf { color: #ff0; font-size: 1.2em; font-family: monospace; }
        #status { color: #888; font-size: 0.9em; margin-top: 10px; text-transform: uppercase; letter-spacing: 1px; }
        .recording { color: #f00 !important; font-weight: bold; animation: blink 1s infinite; }
        @keyframes blink { 50% { opacity: 0.5; } }
    </style>
</head>

<body>
    <h2>FSL Dynamic Gesture Recognition</h2>

    <div id="wrap">
        <video id="video" width="640" height="480" autoplay playsinline></video>
        <canvas id="canvas" width="640" height="480"></canvas>
    </div>

    <div class="ui-panel">
        <div id="pred">INITIALIZING...</div>
        <div id="conf">CONF: 0.00</div>
        <div id="status">Loading Assets...</div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/hands.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/drawing_utils@0.3/drawing_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils@0.3/camera_utils.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js"></script>

    <script>
        /* ================= CONFIG (Matches Python) ================= */
        const SEQ_LEN = 50;
        const VECTOR_LEN = 89;
        const CONF_THRESHOLD = 0.65;
        const NO_MOTION_REQUIRED = 8;
        const COOLDOWN_FRAMES = 15;
        const BASE_MOTION_NOISE = 0.003;
        const SMOOTHING_ALPHA = 0.6;

        let session = null, inputName = null, outputName = null;
        let LABELS = [];
        let normMean = null, normStd = null;

        let sequence = [];
        let prevLm = null;
        let smoothLm = null;
        let noMotionCount = 0;
        let cooldown = 0;

        /* ================= MATH UTILS ================= */
        function softmax(arr) {
            const m = Math.max(...arr);
            const ex = arr.map(v => Math.exp(v - m));
            const s = ex.reduce((a, b) => a + b, 0);
            return ex.map(v => v / s);
        }

        function computeFingerStates(p) {
            // p is flat [x0, y0, x1, y1...] 
            // We need Y coordinates of specific joints (indices 1, 3, 5, 7, 9... of the flat array)
            // p[9] is Tip Y, p[5] is PIP Y for Index finger, etc.
            return [
                p[4*2+1]  < p[3*2+1]  ? 1.0 : 0.0, // Thumb
                p[8*2+1]  < p[6*2+1]  ? 1.0 : 0.0, // Index
                p[12*2+1] < p[10*2+1] ? 1.0 : 0.0, // Middle
                p[16*2+1] < p[14*2+1] ? 1.0 : 0.0, // Ring
                p[20*2+1] < p[18*2+1] ? 1.0 : 0.0  // Pinky
            ];
        }

        /* ================= ASSET LOADING ================= */
        async function loadAssets() {
            try {
                ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";

                const [mRes, sRes, lRes] = await Promise.all([
                    fetch(`./norm_mean.json?v=${Date.now()}`),
                    fetch(`./norm_std.json?v=${Date.now()}`),
                    fetch(`./labels.json?v=${Date.now()}`)
                ]);

                normMean = await mRes.json();
                normStd = await sRes.json();
                LABELS = await lRes.json();

                session = await ort.InferenceSession.create("./gesture_lstm.onnx");
                inputName = session.inputNames[0];
                outputName = session.outputNames[0];

                document.getElementById("status").innerText = "Ready - Show hand to begin";
                document.getElementById("pred").innerText = "WAITING";
            } catch (e) {
                document.getElementById("status").innerText = "Error Loading Assets";
                console.error(e);
            }
        }

        /* ================= MEDIAPIPE LOOP ================= */
        const video = document.getElementById("video");
        const canvas = document.getElementById("canvas");
        const ctx = canvas.getContext("2d");

        const hands = new Hands({
            locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${f}`
        });

        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7
        });

        hands.onResults(res => {
            ctx.clearRect(0, 0, 640, 480);
            if (!session || !normMean) return;
            if (cooldown > 0) cooldown--;

            if (!res.multiHandLandmarks || res.multiHandLandmarks.length === 0) {
                prevLm = smoothLm = null;
                noMotionCount = 0;
                return;
            }

            const hand = res.multiHandLandmarks[0];
            drawConnectors(ctx, hand, HAND_CONNECTIONS, {color: "#0f0", lineWidth: 3});
            drawLandmarks(ctx, hand, {color: "#fff", radius: 2});

            // 1. Flatten Landmarks (42 features)
            const lmFlat = [];
            for (let i = 0; i < 21; i++) {
                lmFlat.push(hand[i].x, hand[i].y);
            }

            // 2. EMA Smoothing
            if (smoothLm === null) {
                smoothLm = [...lmFlat];
            } else {
                smoothLm = smoothLm.map((v, i) => SMOOTHING_ALPHA * v + (1 - SMOOTH_ALPHA) * lmFlat[i]);
            }

            // 3. Velocity & Motion
            let vel = new Array(42).fill(0);
            let motion = 0;
            if (prevLm !== null) {
                vel = smoothLm.map((v, i) => v - prevLm[i]);
                const absSum = vel.reduce((a, b) => a + Math.abs(b), 0);
                motion = absSum / 42;
            }

            const fstates = computeFingerStates(smoothLm);
            const semanticFrame = [...smoothLm, ...vel, ...fstates];
            prevLm = [...smoothLm];

            // 4. Recording Logic
            const statusEl = document.getElementById("status");
            if (cooldown === 0) {
                if (motion > BASE_MOTION_NOISE) {
                    sequence.push(semanticFrame);
                    noMotionCount = 0;
                    statusEl.innerText = "RECORDING...";
                    statusEl.className = "status recording";
                } else {
                    noMotionCount++;
                }
            }

            // 5. Trigger Inference
            if (noMotionCount >= NO_MOTION_REQUIRED && sequence.length >= 15) {
                statusEl.innerText = "PROCESSING...";
                statusEl.className = "status";

                // Padding logic: repeat last frame
                let seqArr = [...sequence];
                const lastFrame = seqArr[seqArr.length - 1];
                while (seqArr.length < SEQ_LEN) {
                    seqArr.push([...lastFrame]);
                }
                if (seqArr.length > SEQ_LEN) {
                    seqArr = seqArr.slice(-SEQ_LEN);
                }

                // Normalization
                const normalized = seqArr.map(frame => 
                    frame.map((v, i) => (v - normMean[i]) / (normStd[i] + 1e-6))
                );

                const tensor = new ort.Tensor("float32", Float32Array.from(normalized.flat()), [1, SEQ_LEN, VECTOR_LEN]);

                session.run({ [inputName]: tensor }).then(out => {
                    const probs = softmax(Array.from(out[outputName].data));
                    const idx = probs.indexOf(Math.max(...probs));
                    const confidence = probs[idx];

                    if (confidence >= CONF_THRESHOLD) {
                        document.getElementById("pred").innerText = LABELS[idx];
                        document.getElementById("conf").innerText = `CONF: ${confidence.toFixed(2)}`;
                    } else {
                        document.getElementById("pred").innerText = "UNCERTAIN";
                        document.getElementById("conf").innerText = `CONF: ${confidence.toFixed(2)}`;
                    }
                });

                // Reset
                sequence = [];
                noMotionCount = 0;
                cooldown = COOLDOWN_FRAMES;
            }
        });

        const camera = new Camera(video, {
            onFrame: async () => { await hands.send({image: video}); },
            width: 640, height: 480
        });

        loadAssets().then(() => camera.start());
    </script>
</body>
</html>