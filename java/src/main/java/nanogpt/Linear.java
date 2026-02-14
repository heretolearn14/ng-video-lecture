package nanogpt;

import java.util.*;

/**
 * Linear (fully connected) layer: out = x @ W^T + b
 * Mirrors torch.nn.Linear.
 */
public class Linear extends Module {

    public final Tensor weight; // (outFeatures, inFeatures)
    public final Tensor bias;   // (outFeatures,) or null
    private final int inFeatures;
    private final int outFeatures;
    private final boolean hasBias;

    public Linear(int inFeatures, int outFeatures, boolean bias, Random rng) {
        this.inFeatures = inFeatures;
        this.outFeatures = outFeatures;
        this.hasBias = bias;

        // Initialize weight with normal(0, 0.02)
        this.weight = Tensor.randn(rng, true, outFeatures, inFeatures);
        for (int i = 0; i < weight.data.length; i++) weight.data[i] *= 0.02f;

        if (bias) {
            this.bias = Tensor.zeros(true, outFeatures);
        } else {
            this.bias = null;
        }
    }

    public Linear(int inFeatures, int outFeatures, Random rng) {
        this(inFeatures, outFeatures, true, rng);
    }

    @Override
    public Tensor forward(Tensor x) {
        // x shape: (..., inFeatures)
        // We need to do x @ W^T which is x @ weight.T
        // For 2D: (N, in) @ (in, out) -> (N, out)
        // For 3D: (B, T, in) @ (in, out) -> (B, T, out)

        // Reshape weight from (out, in) to (in, out) for matmul
        Tensor wT = weight.transpose(0, 1); // (inFeatures, outFeatures)

        Tensor out;
        if (x.ndim() == 2) {
            out = x.matmul(wT); // (N, out)
        } else if (x.ndim() == 3) {
            // Batch matmul: reshape to 2D, matmul, reshape back
            int B = x.shape[0], T = x.shape[1];
            Tensor x2d = x.view(B * T, inFeatures);
            Tensor out2d = x2d.matmul(wT); // (B*T, out)
            out = out2d.view(B, T, outFeatures);
        } else {
            throw new IllegalArgumentException("Linear supports 2D and 3D input, got " + x.ndim() + "D");
        }

        if (hasBias) {
            out = out.add(bias); // broadcast bias
        }
        return out;
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        params.add(weight);
        if (hasBias) params.add(bias);
        return params;
    }
}
