package nanogpt;

import java.util.*;

/**
 * Base class for all neural network modules, mirroring torch.nn.Module.
 * Manages parameters and provides train/eval mode switching.
 */
public abstract class Module {

    protected boolean training = true;

    /** Forward pass. Subclasses implement this. */
    public abstract Tensor forward(Tensor x);

    /** Collect all learnable parameters from this module and its children. */
    public List<Tensor> parameters() {
        return new ArrayList<>();
    }

    /** Collect all child modules. */
    public List<Module> children() {
        return new ArrayList<>();
    }

    /** Set training mode. */
    public void train() {
        this.training = true;
        for (Module child : children()) child.train();
    }

    /** Set evaluation mode. */
    public void eval() {
        this.training = false;
        for (Module child : children()) child.eval();
    }

    /** Zero all parameter gradients. */
    public void zeroGrad() {
        for (Tensor p : parameters()) {
            p.zeroGrad();
        }
    }

    /** Count total number of parameters. */
    public long numParameters() {
        long total = 0;
        for (Tensor p : parameters()) {
            total += p.numel();
        }
        return total;
    }

    /** Initialize weights: normal(0, 0.02) for weights, zeros for biases. */
    public void initWeights(Random rng) {
        for (Tensor p : parameters()) {
            for (int i = 0; i < p.data.length; i++) {
                p.data[i] = (float) (rng.nextGaussian() * 0.02);
            }
        }
    }
}
