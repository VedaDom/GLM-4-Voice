class StreamProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array(0);
        this.bufferSize = 2048; // Processing buffer size
        this.fadeInSamples = 128; // Number of samples for fade in
        this.fadeOutSamples = 128; // Number of samples for fade out
    }

    process(inputs, outputs, parameters) {
        const output = outputs[0];
        const channel = output[0];

        if (this.buffer.length < this.bufferSize) {
            // Fill output buffer with zeros if not enough data
            channel.fill(0);
            return true;
        }

        // Copy data from internal buffer to output
        for (let i = 0; i < channel.length; i++) {
            channel[i] = this.buffer[i];

            // Apply fade in at the start of each chunk
            if (i < this.fadeInSamples) {
                const fadeIn = i / this.fadeInSamples;
                channel[i] *= fadeIn;
            }
            // Apply fade out at the end of each chunk
            else if (i >= channel.length - this.fadeOutSamples) {
                const fadeOut = (channel.length - i) / this.fadeOutSamples;
                channel[i] *= fadeOut;
            }
        }

        // Remove processed data from buffer
        this.buffer = this.buffer.slice(channel.length);

        return true;
    }

    // Method to add new audio data to the buffer
    addToBuffer(newData) {
        const newBuffer = new Float32Array(this.buffer.length + newData.length);
        newBuffer.set(this.buffer);
        newBuffer.set(newData, this.buffer.length);
        this.buffer = newBuffer;
    }
}

registerProcessor('stream-processor', StreamProcessor);