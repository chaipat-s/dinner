"use strict";
/* eslint-disable no-undef */
class WonderfulPlaybackProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.active = true;
        this.queue = [];
        this.offset = 0;
        this.marks = [];
        this.written = 0;
        this.read = 0;
        this.port.onmessage = ({ data }) => {
            if (data?.type === "push") {
                const samples = new Float32Array(data.samples);
                this.queue.push(samples);
                this.written += samples.length;
            }
            else if (data?.type === "mark") {
                this.marks.push({ id: data.id, position: this.written });
            }
            else if (data?.type === "clear") {
                this.queue = [];
                this.offset = 0;
                this.marks = [];
                this.written = 0;
                this.read = 0;
            }
            else if (data?.type === "stop")
                this.active = false;
        };
    }
    process(_inputs, outputs) {
        if (!this.active)
            return false;
        const output = outputs[0]?.[0];
        if (!output)
            return true;
        output.fill(0);
        for (let i = 0; i < output.length && this.queue.length > 0; i++) {
            const head = this.queue[0];
            output[i] = head[this.offset++] || 0;
            this.read++;
            if (this.offset === head.length) {
                this.queue.shift();
                this.offset = 0;
            }
        }
        const pending = [];
        for (const mark of this.marks) {
            if (this.read >= mark.position)
                this.port.postMessage({ type: "markReached", id: mark.id });
            else
                pending.push(mark);
        }
        this.marks = pending;
        return true;
    }
}
registerProcessor("wonderful-playback", WonderfulPlaybackProcessor);
