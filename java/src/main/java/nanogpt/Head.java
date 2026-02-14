package nanogpt;

import java.util.*;

/**
 * One head of self-attention.
 * Mirrors the Head class from gpt.py.
 */
public class Head extends Module {

    private final Linear key;
    private final Linear query;
    private final Linear value;
    private final float[] tril; // lower triangular mask (blockSize * blockSize)
    private final int blockSize;
    private final int headSize;
    private final float dropoutProb;
    private final Random rng;

    public Head(int nEmbd, int headSize, int blockSize, float dropout, Random rng) {
        this.headSize = headSize;
        this.blockSize = blockSize;
        this.dropoutProb = dropout;
        this.rng = rng;

        this.key = new Linear(nEmbd, headSize, false, rng);
        this.query = new Linear(nEmbd, headSize, false, rng);
        this.value = new Linear(nEmbd, headSize, false, rng);

        // Create lower triangular mask
        this.tril = new float[blockSize * blockSize];
        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j <= i; j++) {
                tril[i * blockSize + j] = 1.0f;
            }
        }
    }

    @Override
    public Tensor forward(Tensor x) {
        // x: (B, T, C)
        int B = x.shape[0], T = x.shape[1], C = x.shape[2];

        Tensor k = key.forward(x);   // (B, T, headSize)
        Tensor q = query.forward(x); // (B, T, headSize)

        // Compute attention scores: q @ k^T * scale
        Tensor kT = k.transpose(1, 2); // (B, headSize, T)
        Tensor wei = q.matmul(kT);     // (B, T, T)
        float scale = (float) Math.pow(headSize, -0.5);
        wei = wei.mul(scale);

        // Apply causal mask: use only the T x T portion of the full tril
        float[] mask = new float[T * T];
        for (int i = 0; i < T; i++) {
            for (int j = 0; j < T; j++) {
                mask[i * T + j] = tril[i * blockSize + j];
            }
        }
        wei = wei.maskedFill(mask, Float.NEGATIVE_INFINITY);

        // Softmax
        wei = wei.softmax();

        // Dropout
        boolean[] dropMask = null;
        if (training && dropoutProb > 0) {
            dropMask = new boolean[B * T * T];
            for (int i = 0; i < dropMask.length; i++) {
                dropMask[i] = rng.nextFloat() >= dropoutProb;
            }
        }
        wei = wei.dropout(dropMask, dropoutProb);

        // Weighted aggregation of values
        Tensor v = value.forward(x); // (B, T, headSize)
        Tensor out = wei.matmul(v);  // (B, T, headSize)
        return out;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        params.addAll(key.parameters());
        params.addAll(query.parameters());
        params.addAll(value.parameters());
        return params;
    }

    @Override
    public List<Module> children() {
        return List.of(key, query, value);
    }
}
