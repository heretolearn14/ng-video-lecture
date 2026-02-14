package nanogpt;

import java.util.*;

/**
 * Multiple heads of self-attention in parallel.
 * Mirrors the MultiHeadAttention class from gpt.py.
 */
public class MultiHeadAttention extends Module {

    private final List<Head> heads;
    private final Linear proj;
    private final float dropoutProb;
    private final Random rng;

    public MultiHeadAttention(int numHeads, int headSize, int nEmbd, int blockSize,
                              float dropout, Random rng) {
        this.dropoutProb = dropout;
        this.rng = rng;

        this.heads = new ArrayList<>();
        for (int i = 0; i < numHeads; i++) {
            heads.add(new Head(nEmbd, headSize, blockSize, dropout, rng));
        }
        this.proj = new Linear(headSize * numHeads, nEmbd, rng);
    }

    @Override
    public Tensor forward(Tensor x) {
        // Run each head and concatenate along last dimension
        List<Tensor> headOutputs = new ArrayList<>();
        for (Head head : heads) {
            headOutputs.add(head.forward(x));
        }
        Tensor concatenated = Tensor.cat(headOutputs, -1); // (B, T, headSize * numHeads)

        // Project back
        Tensor out = proj.forward(concatenated); // (B, T, nEmbd)

        // Dropout
        boolean[] dropMask = null;
        if (training && dropoutProb > 0) {
            dropMask = new boolean[out.numel()];
            for (int i = 0; i < dropMask.length; i++) {
                dropMask[i] = rng.nextFloat() >= dropoutProb;
            }
        }
        out = out.dropout(dropMask, dropoutProb);
        return out;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        for (Head head : heads) params.addAll(head.parameters());
        params.addAll(proj.parameters());
        return params;
    }

    @Override
    public List<Module> children() {
        List<Module> ch = new ArrayList<>(heads);
        ch.add(proj);
        return ch;
    }
}
