class VoicePlayer {
    constructor(ctx) {
        this.ctx = ctx;
        this.next = 0;
        this.sources = [];
        this.leadTime = 0.08; // 80ms jitter buffer: prevents under-runs between arriving network chunks
    }

    enqueue(ab) {
        if (!ab || ab.byteLength === 0) return;

        const pcm = new Int16Array(ab, 0, ab.byteLength >> 1);
        const f32 = new Float32Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) {
            f32[i] = pcm[i] / 32768;
        }

        const buf = this.ctx.createBuffer(1, f32.length, 44100);
        buf.copyToChannel(f32, 0);

        const src = this.ctx.createBufferSource();
        src.buffer = buf;
        src.connect(this.ctx.destination);

        const now = this.ctx.currentTime;
        // If buffer depleted (gap/lag), re-anchor playback with a small lead-time cushion
        if (this.next < now) {
            this.next = now + this.leadTime;
        }

        src.start(this.next);
        this.next += buf.duration;
        this.sources.push(src);

        src.onended = () => {
            const idx = this.sources.indexOf(src);
            if (idx !== -1) {
                this.sources.splice(idx, 1);
            }
        };
    }

    stop() {
        this.sources.forEach((s) => {
            try {
                s.stop();
            } catch (e) {}
        });
        this.sources = [];
        this.next = 0;
    }
}

async function startVoice(socket) {
    // 1. Playback Context: native 44.1kHz context matching Cartesia audio output
    const playCtx = new (window.AudioContext || window.webkitAudioContext)();
    await playCtx.resume();
    window.voicePlayer = new VoicePlayer(playCtx);

    // 2. Capture Context: dedicated context for microphone capture
    const captureCtx = new (window.AudioContext || window.webkitAudioContext)();
    await captureCtx.resume();

    // Use absolute URL resolution so worklet path resolves correctly on Vercel
    const workletUrl = new URL("pcm-worklet.js", window.location.href).href;
    await captureCtx.audioWorklet.addModule(workletUrl);

    const media = await navigator.mediaDevices.getUserMedia({
        audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
        },
    });

    const src = captureCtx.createMediaStreamSource(media);
    const node = new AudioWorkletNode(captureCtx, "pcm-capture");

    // Connect analyzer to visually bounce mic indicator bars with real speech volume
    const analyser = captureCtx.createAnalyser();
    analyser.fftSize = 64;
    src.connect(analyser);

    const micIndicator = document.getElementById('mic-indicator');
    const bars = micIndicator ? micIndicator.querySelectorAll('span') : [];
    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    function updateVolumeVisualizer() {
        if (!window.voiceActive) return;
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            sum += dataArray[i];
        }
        const avg = sum / dataArray.length;
        if (bars.length >= 3) {
            bars[0].style.height = `${Math.max(4, Math.min(18, (avg / 255) * 26 + 4))}px`;
            bars[1].style.height = `${Math.max(4, Math.min(18, (avg / 255) * 32 + 6))}px`;
            bars[2].style.height = `${Math.max(4, Math.min(18, (avg / 255) * 22 + 4))}px`;
        }
        requestAnimationFrame(updateVolumeVisualizer);
    }
    window.voiceActive = true;
    requestAnimationFrame(updateVolumeVisualizer);

    let packetsSent = 0;
    node.port.onmessage = (e) => {
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(e.data);
            packetsSent++;
            if (packetsSent === 1) {
                console.log("[Voice] First audio packet sent to server successfully");
            }
        }
    };

    const mute = captureCtx.createGain();
    mute.gain.value = 0;
    src.connect(node);
    node.connect(mute);
    mute.connect(captureCtx.destination);

    return true;
}
