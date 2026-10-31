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
    // Specify 16000Hz directly so the browser native audio hardware/pipeline resamples cleanly
    const ctx = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000,
        latencyHint: "interactive"
    });
    await ctx.resume();

    // Use absolute URL resolution so worklet path resolves correctly on Vercel
    const workletUrl = new URL("pcm-worklet.js", window.location.href).href;
    await ctx.audioWorklet.addModule(workletUrl);

    const media = await navigator.mediaDevices.getUserMedia({
        audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
        },
    });

    const src = ctx.createMediaStreamSource(media);
    const node = new AudioWorkletNode(ctx, "pcm-capture");

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

    const mute = ctx.createGain();
    mute.gain.value = 0;
    src.connect(node);
    node.connect(mute);
    mute.connect(ctx.destination);

    window.voicePlayer = new VoicePlayer(ctx);
    return true;
}
