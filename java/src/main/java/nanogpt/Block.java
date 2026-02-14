package nanogpt;

import java.util.*;

/**
 * Transformer block: communication (self-attention) followed by computation (feed-forward).
 * Uses pre-norm architecture with residual connections.
 * Mirrors the Block class from gpt.py.
 */
public class Block extends Module {

    private final MultiHeadAttention sa;
    private final FeedForward ffwd;
    private final LayerNorm ln1;
    private final LayerNorm ln2;

    public Block(int nEmbd, int nHead, int blockSize, float dropout, Random rng) {
        int headSize = nEmbd / nHead;
        this.sa = new MultiHeadAttention(nHead, headSize, nEmbd, blockSize, dropout, rng);
        this.ffwd = new FeedForward(nEmbd, dropout, rng);
        this.ln1 = new LayerNorm(nEmbd);
        this.ln2 = new LayerNorm(nEmbd);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x + self_attention(layer_norm(x))
        Tensor attnOut = sa.forward(ln1.forward(x));
        x = x.add(attnOut);
        // x + feed_forward(layer_norm(x))
        Tensor ffwdOut = ffwd.forward(ln2.forward(x));
        x = x.add(ffwdOut);
        return x;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        params.addAll(sa.parameters());
        params.addAll(ffwd.parameters());
        params.addAll(ln1.parameters());
        params.addAll(ln2.parameters());
        return params;
    }

    @Override
    public List<Module> children() {
        return List.of(sa, ffwd, ln1, ln2);
    }
}
