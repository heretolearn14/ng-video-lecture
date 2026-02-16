package nanogpt;

import java.util.*;

/**
 * Simple bigram language model.
 * Each token directly reads off the logits for the next token from a lookup table.
 * Mirrors the BigramLanguageModel from bigram.py.
 */
public class BigramLanguageModel extends Module {

    private final Embedding tokenEmbeddingTable;
    private final int vocabSize;

    public BigramLanguageModel(int vocabSize, Random rng) {
        this.vocabSize = vocabSize;
        this.tokenEmbeddingTable = new Embedding(vocabSize, vocabSize, rng);
    }

    /**
     * Forward pass.
     * @param idx input token indices, flat array of shape (B*T)
     * @param idxShape shape of idx, e.g. {B, T}
     * @param targets target token indices, flat array of shape (B*T), or null for generation
     * @return [logits, loss] where loss may be null
     */
    public Tensor[] forward(int[] idx, int[] idxShape, int[] targets) {
        // Look up embeddings: (B, T, vocabSize)
        Tensor logits = tokenEmbeddingTable.forward(idx, idxShape);

        Tensor loss = null;
        if (targets != null) {
            int B = idxShape[0], T = idxShape[1];
            // Reshape logits to (B*T, C) for cross entropy
            Tensor logits2d = logits.view(B * T, vocabSize);
            loss = Tensor.crossEntropy(logits2d, targets);
        }

        return new Tensor[]{logits, loss};
    }

    /**
     * Generate new tokens autoregressively.
     * @param idx starting context, shape (1, 1) typically
     * @param maxNewTokens number of tokens to generate
     * @param rng random number generator
     * @return generated token indices
     */
    public int[] generate(int[] idx, int maxNewTokens, Random rng) {
        List<Integer> tokens = new ArrayList<>();
        for (int i : idx) tokens.add(i);

        // No gradient tracking during generation
        Tensor.noGrad = true;
        try {
            for (int i = 0; i < maxNewTokens; i++) {
                int[] currentIdx = tokens.stream().mapToInt(Integer::intValue).toArray();
                int[] idxShape = {1, currentIdx.length};

                Tensor[] result = forward(currentIdx, idxShape, null);
                Tensor logits = result[0]; // (1, T, vocabSize)

                // Focus on last time step
                Tensor lastLogits = logits.selectAlongDim(1, -1); // (1, vocabSize)
                // Squeeze to 1D
                Tensor probs = lastLogits.view(vocabSize).softmax();

                // Sample
                int nextToken = probs.multinomial(rng);
                tokens.add(nextToken);
            }
        } finally {
            Tensor.noGrad = false;
        }

        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException("Use forward(int[], int[], int[]) instead");
    }

    @Override
    public List<Tensor> parameters() {
        return tokenEmbeddingTable.parameters();
    }

    @Override
    public List<Module> children() {
        return List.of(tokenEmbeddingTable);
    }
}
