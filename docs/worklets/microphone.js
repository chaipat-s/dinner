"use strict";
/* eslint-disable no-undef */
class WonderfulMicrophoneProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.active = true;
        this.muted = false;
        this.chunkSize = Math.max(128, Math.round(sampleRate * 0.02));
        this.buffer = new Float32Array(this.chunkSize);
        this.index = 0;
        this.port.onmessage = ({ data }) => {
            if (data?.type === "stop")
                this.active = false;
            if (data?.type === "mute") {
                const muted = data.muted === true;
                if (muted && !this.muted && this.index > 0) {
                    const chunk = this.buffer.slice(0, this.index);
                    this.port.postMessage({ type: "chunk", samples: chunk.buffer, sampleRate }, [chunk.buffer]);
                    this.buffer = new Float32Array(this.chunkSize);
                    this.index = 0;
                }
                this.muted = muted;
            }
        };
    }
    process(inputs) {
        if (!this.active)
            return false;
        const input = inputs[0]?.[0];
        if (!input)
            return true;
        for (let offset = 0; offset < input.length;) {
            const count = Math.min(this.chunkSize - this.index, input.length - offset);
            if (this.muted)
                this.buffer.fill(0, this.index, this.index + count);
            else
                this.buffer.set(input.subarray(offset, offset + count), this.index);
            this.index += count;
            offset += count;
            if (this.index === this.chunkSize) {
                const chunk = this.buffer;
                this.port.postMessage({ type: "chunk", samples: chunk.buffer, sampleRate }, [chunk.buffer]);
                this.buffer = new Float32Array(this.chunkSize);
                this.index = 0;
            }
        }
        return true;
    }
}
registerProcessor("wonderful-microphone", WonderfulMicrophoneProcessor);
