class PCMCapture extends AudioWorkletProcessor {
    constructor() {
        super();
        this.sampleRate = sampleRate;
        this.ratio = sampleRate / 16000;
        this.phase = 0;
        this.acc = 0;
        this.cnt = 0;
        this.out = new Int16Array(512);
        this.n = 0;
    }

    process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (!ch) return true;

        if (this.sampleRate === 16000) {
            // Direct 1:1 copy when AudioContext is configured to 16kHz (zero interpolation jitter)
            for (let i = 0; i < ch.length; i++) {
                const s = Math.max(-1, Math.min(1, ch[i]));
                this.out[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
                if (this.n === 512) {
                    this.port.postMessage(this.out.buffer.slice(0));
                    this.n = 0;
                }
            }
            return true;
        }

        // Resample fallback when AudioContext forced to 44.1k/48k
        for (let i = 0; i < ch.length; i++) {
            this.acc += ch[i];
            this.cnt++;
            this.phase += 1;

            if (this.phase >= this.ratio) {
                this.phase -= this.ratio;
                const s = Math.max(-1, Math.min(1, this.acc / this.cnt));
                this.out[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
                this.acc = 0;
                this.cnt = 0;

                if (this.n === 512) {
                    this.port.postMessage(this.out.buffer.slice(0));
                    this.n = 0;
                }
            }
        }
        return true;
    }
}

registerProcessor("pcm-capture", PCMCapture);
