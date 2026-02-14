package nanogpt;

import java.util.*;

/**
 * Layer Normalization along the last dimension.
 * Mirrors torch.nn.LayerNorm.
 */
public class LayerNorm extends Module {

    public final Tensor gamma; // scale parameter (lastDim,)
    public final Tensor beta;  // shift parameter (lastDim,)
    private final int normalizedSize;
    private final float eps;

    public LayerNorm(int normalizedSize) {
        this(normalizedSize, 1e-5f);
    }

    public LayerNorm(int normalizedSize, float eps) {
        this.normalizedSize = normalizedSize;
        this.eps = eps;

        // Initialize gamma = 1, beta = 0
        float[] gammaData = new float[normalizedSize];
        Arrays.fill(gammaData, 1.0f);
        this.gamma = new Tensor(gammaData, new int[]{normalizedSize}, true);
        this.beta = Tensor.zeros(true, normalizedSize);
    }

    @Override
    public Tensor forward(Tensor x) {
        return Tensor.layerNorm(x, gamma, beta, eps);
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(gamma, beta);
    }
}
