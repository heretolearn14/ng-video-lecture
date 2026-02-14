package nanogpt;

import java.util.*;

/**
 * Embedding layer: a lookup table mapping integer indices to dense vectors.
 * Mirrors torch.nn.Embedding.
 */
public class Embedding extends Module {

    public final Tensor weight; // (numEmbeddings, embeddingDim)
    private final int numEmbeddings;
    private final int embeddingDim;

    public Embedding(int numEmbeddings, int embeddingDim, Random rng) {
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;

        // Initialize with normal(0, 0.02)
        this.weight = Tensor.randn(rng, true, numEmbeddings, embeddingDim);
        for (int i = 0; i < weight.data.length; i++) weight.data[i] *= 0.02f;
    }

    /**
     * Look up embeddings for the given indices.
     * @param indices flat int array of token indices
     * @param indicesShape shape of the indices (e.g., {B, T})
     * @return Tensor of shape (*indicesShape, embeddingDim)
     */
    public Tensor forward(int[] indices, int[] indicesShape) {
        return weight.embeddingLookup(indices, indicesShape);
    }

    @Override
    public Tensor forward(Tensor x) {
        // Convert tensor to int indices
        int[] indices = new int[x.numel()];
        for (int i = 0; i < indices.length; i++) {
            indices[i] = (int) x.data[i];
        }
        return forward(indices, x.shape);
    }

    @Override
    public List<Tensor> parameters() {
        return List.of(weight);
    }
}
