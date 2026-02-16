package nanogpt;

import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Main entry point for nanoGPT Java implementation.
 * Supports both bigram and GPT model training.
 * Mirrors the training loops from bigram.py and gpt.py.
 *
 * Usage:
 *   java nanogpt.Main bigram [input.txt]    - Train bigram model
 *   java nanogpt.Main gpt [input.txt]       - Train GPT model
 */
public class Main {

    public static void main(String[] args) throws IOException {
        String mode = args.length > 0 ? args[0] : "bigram";
        String inputFile = args.length > 1 ? args[1] : "input.txt";

        System.out.println("nanoGPT Java - " + mode + " model");
        System.out.println("Reading input from: " + inputFile);

        // Read input text
        String text = Files.readString(Path.of(inputFile));
        System.out.println("Input length: " + text.length() + " characters");

        // Create tokenizer
        Tokenizer tokenizer = new Tokenizer(text);
        int vocabSize = tokenizer.getVocabSize();
        System.out.println("Vocab size: " + vocabSize);

        // Encode the full text
        int[] data = tokenizer.encode(text);

        if (mode.equals("bigram")) {
            trainBigram(data, vocabSize, tokenizer);
        } else if (mode.equals("gpt")) {
            trainGPT(data, vocabSize, tokenizer);
        } else {
            System.err.println("Unknown mode: " + mode + ". Use 'bigram' or 'gpt'.");
        }
    }

    // ---- Bigram Model Training ----

    private static void trainBigram(int[] data, int vocabSize, Tokenizer tokenizer) {
        // Hyperparameters (matching bigram.py)
        int batchSize = 32;
        int blockSize = 8;
        int maxIters = 3000;
        int evalInterval = 300;
        float learningRate = 1e-2f;
        int evalIters = 200;

        Random rng = new Random(1337);

        // Create data loader
        DataLoader dataLoader = new DataLoader(data, blockSize, batchSize, rng);
        System.out.println("Train data size: " + dataLoader.getTrainSize());
        System.out.println("Val data size: " + dataLoader.getValSize());

        // Create model
        BigramLanguageModel model = new BigramLanguageModel(vocabSize, rng);
        List<Tensor> params = model.parameters();
        System.out.println("Parameters: " + countParams(params));

        // Create optimizer
        AdamW optimizer = new AdamW(params, learningRate);

        // Training loop
        int[] shape = dataLoader.getShape();
        for (int iter = 0; iter < maxIters; iter++) {
            // Evaluate loss periodically
            if (iter % evalInterval == 0) {
                float[] losses = estimateLoss(model, dataLoader, evalIters);
                System.out.printf("step %d: train loss %.4f, val loss %.4f%n",
                        iter, losses[0], losses[1]);
            }

            // Get batch
            int[][] batch = dataLoader.getBatch("train");
            int[] xb = batch[0];
            int[] yb = batch[1];

            // Forward pass
            Tensor[] result = model.forward(xb, shape, yb);
            Tensor loss = result[1];

            // Backward pass
            optimizer.zeroGrad();
            loss.backward();

            // Update parameters
            optimizer.step();
        }

        // Generate text
        System.out.println("\n--- Generated Text ---");
        model.eval();
        int[] context = {0}; // start with token 0
        int[] generated = model.generate(context, 500, rng);
        System.out.println(tokenizer.decode(generated));
    }

    // ---- GPT Model Training ----

    private static void trainGPT(int[] data, int vocabSize, Tokenizer tokenizer) {
        // Hyperparameters (matching gpt.py)
        int batchSize = 64;
        int blockSize = 256;
        int maxIters = 5000;
        int evalInterval = 500;
        float learningRate = 3e-4f;
        int evalIters = 200;
        int nEmbd = 384;
        int nHead = 6;
        int nLayer = 6;
        float dropout = 0.2f;

        Random rng = new Random(1337);

        // Create data loader
        DataLoader dataLoader = new DataLoader(data, blockSize, batchSize, rng);
        System.out.println("Train data size: " + dataLoader.getTrainSize());
        System.out.println("Val data size: " + dataLoader.getValSize());

        // Create model
        GPTLanguageModel model = new GPTLanguageModel(
                vocabSize, nEmbd, nHead, nLayer, blockSize, dropout, rng);
        List<Tensor> params = model.parameters();
        System.out.printf("%.3f M parameters%n", countParams(params) / 1e6);

        // Create optimizer
        AdamW optimizer = new AdamW(params, learningRate);

        // Training loop
        int[] shape = dataLoader.getShape();
        long trainStart = System.currentTimeMillis();
        for (int iter = 0; iter < maxIters; iter++) {
            long iterStart = System.currentTimeMillis();

            // Evaluate loss periodically
            if (iter % evalInterval == 0 || iter == maxIters - 1) {
                System.out.printf("Evaluating at step %d...%n", iter);
                float[] losses = estimateLoss(model, dataLoader, evalIters);
                System.out.printf("step %d: train loss %.4f, val loss %.4f%n",
                        iter, losses[0], losses[1]);
            }

            // Get batch
            int[][] batch = dataLoader.getBatch("train");
            int[] xb = batch[0];
            int[] yb = batch[1];

            // Forward pass
            Tensor[] result = model.forward(xb, shape, yb);
            Tensor loss = result[1];

            // Backward pass
            optimizer.zeroGrad();
            loss.backward();

            // Update parameters
            optimizer.step();

            // Heartbeat every 10 iterations
            if ((iter + 1) % 10 == 0) {
                long iterMs = System.currentTimeMillis() - iterStart;
                long totalSec = (System.currentTimeMillis() - trainStart) / 1000;
                System.out.printf("  iter %d/%d | loss %.4f | %dms/iter | %ds elapsed%n",
                        iter + 1, maxIters, loss.item(), iterMs, totalSec);
            }
        }

        // Generate text
        System.out.println("\n--- Generated Text ---");
        model.eval();
        int[] context = {0}; // start with token 0
        int[] generated = model.generate(context, 500, rng);
        System.out.println(tokenizer.decode(generated));
    }

    // ---- Helper Methods ----

    /**
     * Estimate loss on train and val sets by averaging over multiple batches.
     * Mirrors the estimate_loss function from the Python implementation.
     */
    private static float[] estimateLoss(Object model, DataLoader dataLoader, int evalIters) {
        float[] out = new float[2];
        String[] splits = {"train", "val"};
        int[] shape = dataLoader.getShape();

        // Set eval mode — disable gradient tracking (equivalent of @torch.no_grad())
        if (model instanceof Module m) m.eval();
        Tensor.noGrad = true;

        try {
            for (int s = 0; s < 2; s++) {
                float totalLoss = 0;
                long splitStart = System.currentTimeMillis();
                for (int k = 0; k < evalIters; k++) {
                    int[][] batch = dataLoader.getBatch(splits[s]);
                    int[] xb = batch[0];
                    int[] yb = batch[1];

                    Tensor[] result;
                    if (model instanceof BigramLanguageModel bm) {
                        result = bm.forward(xb, shape, yb);
                    } else {
                        result = ((GPTLanguageModel) model).forward(xb, shape, yb);
                    }
                    totalLoss += result[1].item();

                    if ((k + 1) % 50 == 0) {
                        long elapsed = System.currentTimeMillis() - splitStart;
                        System.out.printf("  [eval %s] %d/%d batches (%.1fs)%n",
                                splits[s], k + 1, evalIters, elapsed / 1000.0);
                    }
                }
                out[s] = totalLoss / evalIters;
            }
        } finally {
            // Always restore gradient tracking and train mode
            Tensor.noGrad = false;
            if (model instanceof Module m) m.train();
        }

        return out;
    }

    private static long countParams(List<Tensor> params) {
        long total = 0;
        for (Tensor p : params) total += p.numel();
        return total;
    }
}
