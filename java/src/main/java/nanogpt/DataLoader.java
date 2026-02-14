package nanogpt;

import java.util.*;

/**
 * Data loader for creating mini-batches of sequential data.
 * Mirrors the get_batch function from the Python implementation.
 */
public class DataLoader {

    private final int[] trainData;
    private final int[] valData;
    private final int blockSize;
    private final int batchSize;
    private final Random rng;

    public DataLoader(int[] data, int blockSize, int batchSize, Random rng) {
        this.blockSize = blockSize;
        this.batchSize = batchSize;
        this.rng = rng;

        // Split: 90% train, 10% validation
        int n = (int) (0.9 * data.length);
        this.trainData = Arrays.copyOfRange(data, 0, n);
        this.valData = Arrays.copyOfRange(data, n, data.length);
    }

    /**
     * Generate a random mini-batch of input/target pairs.
     * @param split "train" or "val"
     * @return [x, y] where each is a flat int array of shape (batchSize * blockSize),
     *         plus [xShape, yShape] as {batchSize, blockSize}
     */
    public int[][] getBatch(String split) {
        int[] data = split.equals("train") ? trainData : valData;

        int[] x = new int[batchSize * blockSize];
        int[] y = new int[batchSize * blockSize];

        for (int b = 0; b < batchSize; b++) {
            int i = rng.nextInt(data.length - blockSize);
            for (int t = 0; t < blockSize; t++) {
                x[b * blockSize + t] = data[i + t];
                y[b * blockSize + t] = data[i + t + 1];
            }
        }

        return new int[][]{x, y};
    }

    public int[] getShape() {
        return new int[]{batchSize, blockSize};
    }

    public int getTrainSize() { return trainData.length; }
    public int getValSize() { return valData.length; }
}
