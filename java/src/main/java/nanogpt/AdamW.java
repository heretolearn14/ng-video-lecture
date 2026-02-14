package nanogpt;

import java.util.*;

/**
 * AdamW optimizer with weight decay.
 * Mirrors torch.optim.AdamW.
 */
public class AdamW {

    private final List<Tensor> params;
    private final float lr;
    private final float beta1;
    private final float beta2;
    private final float eps;
    private final float weightDecay;
    private int t; // time step

    // First and second moment estimates for each parameter
    private final float[][] m; // first moment
    private final float[][] v; // second moment

    public AdamW(List<Tensor> params, float lr) {
        this(params, lr, 0.9f, 0.999f, 1e-8f, 0.01f);
    }

    public AdamW(List<Tensor> params, float lr, float beta1, float beta2, float eps, float weightDecay) {
        this.params = params;
        this.lr = lr;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.eps = eps;
        this.weightDecay = weightDecay;
        this.t = 0;

        this.m = new float[params.size()][];
        this.v = new float[params.size()][];
        for (int i = 0; i < params.size(); i++) {
            m[i] = new float[params.get(i).numel()];
            v[i] = new float[params.get(i).numel()];
        }
    }

    /** Perform one optimization step. */
    public void step() {
        t++;
        for (int p = 0; p < params.size(); p++) {
            Tensor param = params.get(p);
            if (param.grad == null) continue;

            for (int i = 0; i < param.data.length; i++) {
                float g = param.grad[i];

                // Update biased first moment estimate
                m[p][i] = beta1 * m[p][i] + (1 - beta1) * g;
                // Update biased second raw moment estimate
                v[p][i] = beta2 * v[p][i] + (1 - beta2) * g * g;

                // Bias-corrected estimates
                float mHat = m[p][i] / (1 - (float) Math.pow(beta1, t));
                float vHat = v[p][i] / (1 - (float) Math.pow(beta2, t));

                // Weight decay (decoupled)
                param.data[i] -= lr * weightDecay * param.data[i];

                // Parameter update
                param.data[i] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
            }
        }
    }

    /** Zero all parameter gradients. */
    public void zeroGrad() {
        for (Tensor param : params) {
            if (param.grad != null) {
                Arrays.fill(param.grad, 0.0f);
            }
        }
    }
}
