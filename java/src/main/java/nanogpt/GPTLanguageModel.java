package nanogpt;

import java.util.*;

/**
 * Full GPT language model with transformer architecture.
 * Mirrors the GPTLanguageModel from gpt.py.
 */
public class GPTLanguageModel extends Module {

    private final Embedding tokenEmbeddingTable;
    private final Embedding positionEmbeddingTable;
    private final List<Block> blocks;
    private final LayerNorm lnF;
    private final Linear lmHead;

    private final int vocabSize;
    private final int nEmbd;
    private final int blockSize;

    public GPTLanguageModel(int vocabSize, int nEmbd, int nHead, int nLayer,
                            int blockSize, float dropout, Random rng) {
        this.vocabSize = vocabSize;
        this.nEmbd = nEmbd;
        this.blockSize = blockSize;

        this.tokenEmbeddingTable = new Embedding(vocabSize, nEmbd, rng);
        this.positionEmbeddingTable = new Embedding(blockSize, nEmbd, rng);

        this.blocks = new ArrayList<>();
        for (int i = 0; i < nLayer; i++) {
            blocks.add(new Block(nEmbd, nHead, blockSize, dropout, rng));
        }

        this.lnF = new LayerNorm(nEmbd);
        this.lmHead = new Linear(nEmbd, vocabSize, rng);
    }

    /**
     * Forward pass.
     * @param idx input token indices, flat array of shape (B*T)
     * @param idxShape shape of idx, e.g. {B, T}
     * @param targets target token indices, flat array of shape (B*T), or null for generation
     * @return [logits, loss] where loss may be null
     */
    public Tensor[] forward(int[] idx, int[] idxShape, int[] targets) {
        int B = idxShape[0], T = idxShape[1];

        // Token embeddings: (B, T, nEmbd)
        Tensor tokEmb = tokenEmbeddingTable.forward(idx, idxShape);

        // Position embeddings: (T, nEmbd)
        int[] positions = new int[T];
        for (int i = 0; i < T; i++) positions[i] = i;
        Tensor posEmb = positionEmbeddingTable.forward(positions, new int[]{T});

        // Add token and position embeddings: (B, T, nEmbd)
        Tensor x = tokEmb.add(posEmb);

        // Pass through transformer blocks
        for (Block block : blocks) {
            x = block.forward(x);
        }

        // Final layer norm
        x = lnF.forward(x); // (B, T, nEmbd)

        // Project to vocabulary size
        Tensor logits = lmHead.forward(x); // (B, T, vocabSize)

        Tensor loss = null;
        if (targets != null) {
            // Reshape for cross entropy
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

        for (int i = 0; i < maxNewTokens; i++) {
            // Crop to last blockSize tokens
            int start = Math.max(0, tokens.size() - blockSize);
            int[] currentIdx = new int[tokens.size() - start];
            for (int j = 0; j < currentIdx.length; j++) {
                currentIdx[j] = tokens.get(start + j);
            }
            int[] idxShape = {1, currentIdx.length};

            Tensor[] result = forward(currentIdx, idxShape, null);
            Tensor logits = result[0]; // (1, T, vocabSize)

            // Focus on last time step
            Tensor lastLogits = logits.selectAlongDim(1, -1); // (1, vocabSize)
            Tensor probs = lastLogits.view(vocabSize).softmax();

            // Sample
            int nextToken = probs.multinomial(rng);
            tokens.add(nextToken);
        }

        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public Tensor forward(Tensor x) {
        throw new UnsupportedOperationException("Use forward(int[], int[], int[]) instead");
    }

    @Override
    public List<Tensor> parameters() {
        List<Tensor> params = new ArrayList<>();
        params.addAll(tokenEmbeddingTable.parameters());
        params.addAll(positionEmbeddingTable.parameters());
        for (Block block : blocks) params.addAll(block.parameters());
        params.addAll(lnF.parameters());
        params.addAll(lmHead.parameters());
        return params;
    }

    @Override
    public List<Module> children() {
        List<Module> ch = new ArrayList<>();
        ch.add(tokenEmbeddingTable);
        ch.add(positionEmbeddingTable);
        ch.addAll(blocks);
        ch.add(lnF);
        ch.add(lmHead);
        return ch;
    }
}
