package nanogpt;

import java.util.*;

/**
 * A simple feed-forward network: Linear -> ReLU -> Linear -> Dropout.
 * Mirrors the FeedFoward class from gpt.py.
 */
public class FeedForward extends Module {

    private final Linear linear1;
    private final Linear linear2;
    private final float dropoutProb;
    private final Random rng;

    public FeedForward(int nEmbd, float dropout, Random rng) {
        this.dropoutProb = dropout;
        this.rng = rng;

        this.linear1 = new Linear(nEmbd, 4 * nEmbd, rng);
        this.linear2 = new Linear(4 * nEmbd, nEmbd, rng);
    }

    @Override
    public Tensor forward(Tensor x) {
        Tensor out = linear1.forward(x);
        out = out.relu();
        out = linear2.forward(out);

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
        params.addAll(linear1.parameters());
        params.addAll(linear2.parameters());
        return params;
    }

    @Override
    public List<Module> children() {
        return List.of(linear1, linear2);
    }
}
