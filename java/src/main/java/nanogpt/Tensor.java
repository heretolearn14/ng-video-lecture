package nanogpt;

import java.util.*;

/**
 * N-dimensional tensor with automatic differentiation support.
 * This is a from-scratch implementation mirroring PyTorch's autograd engine.
 * Each operation creates a new tensor with a backward function that propagates
 * gradients to parent tensors when backward() is called.
 */
public class Tensor {

    public float[] data;
    public float[] grad;
    public int[] shape;
    public boolean requiresGrad;

    // Autograd graph
    Runnable backwardFn;
    List<Tensor> parents = new ArrayList<>();

    /**
     * When true, no autograd graph is built — equivalent to PyTorch's torch.no_grad().
     * Use this during evaluation and generation to avoid OOM from graph accumulation.
     */
    public static boolean noGrad = false;

    // ---- Constructors ----

    public Tensor(float[] data, int[] shape, boolean requiresGrad) {
        this.data = data;
        this.shape = shape;
        this.requiresGrad = requiresGrad;
        if (requiresGrad) {
            this.grad = new float[data.length];
        }
    }

    public Tensor(float[] data, int[] shape) {
        this(data, shape, false);
    }

    // ---- Shape Utilities ----

    public int ndim() { return shape.length; }

    public int size(int dim) {
        if (dim < 0) dim += shape.length;
        return shape[dim];
    }

    public int numel() {
        int n = 1;
        for (int s : shape) n *= s;
        return n;
    }

    /** Compute flat index from multi-dimensional indices. */
    private int flatIndex(int[] indices) {
        int idx = 0;
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= shape[i];
        }
        return idx;
    }

    /** Get strides for this tensor's shape. */
    public int[] strides() {
        int[] s = new int[shape.length];
        s[shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--) {
            s[i] = s[i + 1] * shape[i + 1];
        }
        return s;
    }

    public float get(int... indices) {
        return data[flatIndex(indices)];
    }

    public void set(float val, int... indices) {
        data[flatIndex(indices)] = val;
    }

    // ---- Factory Methods ----

    public static Tensor zeros(boolean requiresGrad, int... shape) {
        int n = 1;
        for (int s : shape) n *= s;
        return new Tensor(new float[n], shape, requiresGrad);
    }

    public static Tensor zeros(int... shape) {
        return zeros(false, shape);
    }

    public static Tensor ones(int... shape) {
        int n = 1;
        for (int s : shape) n *= s;
        float[] data = new float[n];
        Arrays.fill(data, 1.0f);
        return new Tensor(data, shape, false);
    }

    public static Tensor randn(Random rng, boolean requiresGrad, int... shape) {
        int n = 1;
        for (int s : shape) n *= s;
        float[] data = new float[n];
        for (int i = 0; i < n; i++) {
            data[i] = (float) rng.nextGaussian();
        }
        return new Tensor(data, shape, requiresGrad);
    }

    public static Tensor arange(int n) {
        float[] data = new float[n];
        for (int i = 0; i < n; i++) data[i] = i;
        return new Tensor(data, new int[]{n}, false);
    }

    /** Create lower triangular matrix of ones. */
    public static Tensor tril(int size) {
        float[] data = new float[size * size];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j <= i; j++) {
                data[i * size + j] = 1.0f;
            }
        }
        return new Tensor(data, new int[]{size, size}, false);
    }

    // ---- Autograd Engine ----

    /** Backward pass: compute gradients for all tensors in the computation graph. */
    public void backward() {
        // Set gradient of loss to 1.0
        this.grad = new float[this.data.length];
        Arrays.fill(this.grad, 1.0f);

        // Topological sort
        List<Tensor> topo = new ArrayList<>();
        Set<Tensor> visited = new HashSet<>();
        buildTopo(this, topo, visited);
        Collections.reverse(topo);

        // Call backward functions in reverse topological order
        for (Tensor t : topo) {
            if (t.backwardFn != null) {
                t.backwardFn.run();
            }
        }
    }

    private void buildTopo(Tensor t, List<Tensor> topo, Set<Tensor> visited) {
        if (visited.contains(t)) return;
        visited.add(t);
        for (Tensor parent : t.parents) {
            buildTopo(parent, topo, visited);
        }
        topo.add(t);
    }

    public void zeroGrad() {
        if (grad != null) Arrays.fill(grad, 0.0f);
    }

    /** Ensure this tensor has a gradient array allocated. */
    private void ensureGrad() {
        if (grad == null) {
            grad = new float[data.length];
        }
    }

    /** Set up autograd tracking on a result tensor (skipped when noGrad is true). */
    private static void trackAutograd(Tensor result, Runnable backwardFn, Tensor... parents) {
        if (noGrad) {
            result.requiresGrad = false;
            return;
        }
        for (Tensor p : parents) result.parents.add(p);
        result.backwardFn = backwardFn;
    }

    // ---- Tensor Operations ----

    // --- Matrix Multiplication ---

    /**
     * Matrix multiplication supporting 2D and batched 3D tensors.
     * 2D: (M, K) @ (K, N) -> (M, N)
     * 3D: (B, M, K) @ (B, K, N) -> (B, M, N)
     */
    public Tensor matmul(Tensor other) {
        if (this.ndim() == 2 && other.ndim() == 2) {
            return matmul2d(this, other);
        } else if (this.ndim() == 3 && other.ndim() == 3) {
            return matmul3d(this, other);
        } else {
            throw new IllegalArgumentException("matmul supports 2D and 3D tensors, got " +
                    this.ndim() + "D and " + other.ndim() + "D");
        }
    }

    private static Tensor matmul2d(Tensor a, Tensor b) {
        int M = a.shape[0], K = a.shape[1];
        int N = b.shape[1];
        assert b.shape[0] == K : "matmul dimension mismatch";

        float[] out = new float[M * N];
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += a.data[i * K + k] * b.data[k * N + j];
                }
                out[i * N + j] = sum;
            }
        }

        Tensor result = new Tensor(out, new int[]{M, N}, !noGrad && (a.requiresGrad || b.requiresGrad));
        trackAutograd(result, () -> {
            // dA = dC @ B^T
            if (a.requiresGrad) {
                a.ensureGrad();
                for (int i = 0; i < M; i++) {
                    for (int k = 0; k < K; k++) {
                        float sum = 0;
                        for (int j = 0; j < N; j++) {
                            sum += result.grad[i * N + j] * b.data[k * N + j];
                        }
                        a.grad[i * K + k] += sum;
                    }
                }
            }
            // dB = A^T @ dC
            if (b.requiresGrad) {
                b.ensureGrad();
                for (int k = 0; k < K; k++) {
                    for (int j = 0; j < N; j++) {
                        float sum = 0;
                        for (int i = 0; i < M; i++) {
                            sum += a.data[i * K + k] * result.grad[i * N + j];
                        }
                        b.grad[k * N + j] += sum;
                    }
                }
            }
        }, a, b);
        return result;
    }

    private static Tensor matmul3d(Tensor a, Tensor b) {
        int B = a.shape[0], M = a.shape[1], K = a.shape[2];
        int N = b.shape[2];
        assert b.shape[0] == B && b.shape[1] == K : "matmul3d dimension mismatch";

        float[] out = new float[B * M * N];
        for (int batch = 0; batch < B; batch++) {
            int aOff = batch * M * K;
            int bOff = batch * K * N;
            int cOff = batch * M * N;
            for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                    float sum = 0;
                    for (int k = 0; k < K; k++) {
                        sum += a.data[aOff + i * K + k] * b.data[bOff + k * N + j];
                    }
                    out[cOff + i * N + j] = sum;
                }
            }
        }

        Tensor result = new Tensor(out, new int[]{B, M, N}, !noGrad && (a.requiresGrad || b.requiresGrad));
        trackAutograd(result, () -> {
            for (int batch = 0; batch < B; batch++) {
                int aOff = batch * M * K;
                int bOff = batch * K * N;
                int cOff = batch * M * N;
                if (a.requiresGrad) {
                    a.ensureGrad();
                    for (int i = 0; i < M; i++) {
                        for (int k = 0; k < K; k++) {
                            float sum = 0;
                            for (int j = 0; j < N; j++) {
                                sum += result.grad[cOff + i * N + j] * b.data[bOff + k * N + j];
                            }
                            a.grad[aOff + i * K + k] += sum;
                        }
                    }
                }
                if (b.requiresGrad) {
                    b.ensureGrad();
                    for (int k = 0; k < K; k++) {
                        for (int j = 0; j < N; j++) {
                            float sum = 0;
                            for (int i = 0; i < M; i++) {
                                sum += a.data[aOff + i * K + k] * result.grad[cOff + i * N + j];
                            }
                            b.grad[bOff + k * N + j] += sum;
                        }
                    }
                }
            }
        }, a, b);
        return result;
    }

    // --- Element-wise Addition (with broadcasting) ---

    /**
     * Add two tensors. Supports broadcasting:
     * - Same shape: element-wise
     * - (B,T,C) + (T,C): broadcast over batch dimension
     * - (B,T,C) + (C,): broadcast over batch and time dimensions
     */
    public Tensor add(Tensor other) {
        if (Arrays.equals(this.shape, other.shape)) {
            return addSameShape(this, other);
        } else {
            return addBroadcast(this, other);
        }
    }

    private static Tensor addSameShape(Tensor a, Tensor b) {
        float[] out = new float[a.data.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = a.data[i] + b.data[i];
        }
        Tensor result = new Tensor(out, a.shape.clone(), !noGrad && (a.requiresGrad || b.requiresGrad));
        trackAutograd(result, () -> {
            if (a.requiresGrad) {
                a.ensureGrad();
                for (int i = 0; i < a.grad.length; i++) a.grad[i] += result.grad[i];
            }
            if (b.requiresGrad) {
                b.ensureGrad();
                for (int i = 0; i < b.grad.length; i++) b.grad[i] += result.grad[i];
            }
        }, a, b);
        return result;
    }

    private static Tensor addBroadcast(Tensor a, Tensor b) {
        // a is the larger tensor, b is the smaller one to broadcast
        // Determine which is larger
        Tensor large = a.numel() >= b.numel() ? a : b;
        Tensor small = a.numel() >= b.numel() ? b : a;

        float[] out = new float[large.data.length];
        int smallNumel = small.numel();

        // Broadcasting: small is repeated to match large
        for (int i = 0; i < large.data.length; i++) {
            out[i] = large.data[i] + small.data[i % smallNumel];
        }

        Tensor result = new Tensor(out, large.shape.clone(), !noGrad && (large.requiresGrad || small.requiresGrad));
        trackAutograd(result, () -> {
            if (large.requiresGrad) {
                large.ensureGrad();
                for (int i = 0; i < large.grad.length; i++) large.grad[i] += result.grad[i];
            }
            if (small.requiresGrad) {
                small.ensureGrad();
                for (int i = 0; i < large.data.length; i++) {
                    small.grad[i % smallNumel] += result.grad[i];
                }
            }
        }, large, small);
        return result;
    }

    // --- Scalar Multiplication ---

    public Tensor mul(float scalar) {
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = data[i] * scalar;
        }
        Tensor result = new Tensor(out, shape.clone(), !noGrad && requiresGrad);
        trackAutograd(result, () -> {
            if (this.requiresGrad) {
                this.ensureGrad();
                for (int i = 0; i < data.length; i++) {
                    this.grad[i] += result.grad[i] * scalar;
                }
            }
        }, this);
        return result;
    }

    // --- View / Reshape ---

    /**
     * Reshape tensor. Supports one -1 dimension that is inferred.
     * Data is shared (not copied) - gradients flow through directly.
     */
    public Tensor view(int... newShape) {
        int[] resolved = resolveShape(newShape, numel());
        // Share data array, create new tensor with new shape
        Tensor result = new Tensor(this.data, resolved, !noGrad && this.requiresGrad);

        if (noGrad) return result;

        result.parents.add(this);
        if (this.grad != null) {
            // Share grad array directly - no backward function needed since they're the same array
            result.grad = this.grad;
            // Keep parents for topological sort traversal, but no-op backward
            result.backwardFn = null;
        } else {
            result.backwardFn = () -> {
                if (this.requiresGrad) {
                    this.ensureGrad();
                    for (int i = 0; i < data.length; i++) {
                        this.grad[i] += result.grad[i];
                    }
                }
            };
        }
        return result;
    }

    private static int[] resolveShape(int[] shape, int numel) {
        int[] resolved = shape.clone();
        int unknownIdx = -1;
        int known = 1;
        for (int i = 0; i < resolved.length; i++) {
            if (resolved[i] == -1) {
                unknownIdx = i;
            } else {
                known *= resolved[i];
            }
        }
        if (unknownIdx >= 0) {
            resolved[unknownIdx] = numel / known;
        }
        return resolved;
    }

    // --- Transpose ---

    /**
     * Transpose two dimensions.
     * For 3D: transpose(1,2) on (B,T,C) gives (B,C,T).
     */
    public Tensor transpose(int dim0, int dim1) {
        if (dim0 < 0) dim0 += ndim();
        if (dim1 < 0) dim1 += ndim();

        int[] newShape = shape.clone();
        newShape[dim0] = shape[dim1];
        newShape[dim1] = shape[dim0];

        int[] oldStrides = strides();
        int[] newStrides = new int[ndim()];
        for (int i = 0; i < ndim(); i++) newStrides[i] = oldStrides[i];
        newStrides[dim0] = oldStrides[dim1];
        newStrides[dim1] = oldStrides[dim0];

        float[] out = new float[data.length];
        int[] oldIdx = new int[ndim()];
        for (int i = 0; i < data.length; i++) {
            // Convert flat index to multi-dim using newShape and newStrides
            int remaining = i;
            int srcIdx = 0;
            for (int d = 0; d < ndim(); d++) {
                int coord = remaining / strides(newShape, d);
                remaining %= strides(newShape, d);
                srcIdx += coord * newStrides[d];
            }
            out[i] = data[srcIdx];
        }

        final int d0 = dim0, d1 = dim1;
        Tensor result = new Tensor(out, newShape, !noGrad && requiresGrad);
        trackAutograd(result, () -> {
            if (this.requiresGrad) {
                this.ensureGrad();
                int[] ns = newShape;
                for (int i = 0; i < data.length; i++) {
                    int remaining = i;
                    int srcIdx = 0;
                    for (int d = 0; d < ndim(); d++) {
                        int coord = remaining / strides(ns, d);
                        remaining %= strides(ns, d);
                        srcIdx += coord * newStrides[d];
                    }
                    this.grad[srcIdx] += result.grad[i];
                }
            }
        }, this);
        return result;
    }

    private static int strides(int[] shape, int dim) {
        int s = 1;
        for (int i = dim + 1; i < shape.length; i++) s *= shape[i];
        return s;
    }

    // --- Softmax ---

    /**
     * Softmax along the last dimension.
     */
    public Tensor softmax() {
        int lastDim = shape[shape.length - 1];
        int outerSize = data.length / lastDim;

        float[] out = new float[data.length];
        for (int i = 0; i < outerSize; i++) {
            int offset = i * lastDim;
            // Numerical stability: subtract max
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < lastDim; j++) {
                max = Math.max(max, data[offset + j]);
            }
            float sumExp = 0;
            for (int j = 0; j < lastDim; j++) {
                out[offset + j] = (float) Math.exp(data[offset + j] - max);
                sumExp += out[offset + j];
            }
            for (int j = 0; j < lastDim; j++) {
                out[offset + j] /= sumExp;
            }
        }

        Tensor result = new Tensor(out, shape.clone(), !noGrad && requiresGrad);
        trackAutograd(result, () -> {
            if (this.requiresGrad) {
                this.ensureGrad();
                for (int i = 0; i < outerSize; i++) {
                    int offset = i * lastDim;
                    // dx_j = s_j * (dout_j - sum_k(dout_k * s_k))
                    float dot = 0;
                    for (int j = 0; j < lastDim; j++) {
                        dot += result.grad[offset + j] * result.data[offset + j];
                    }
                    for (int j = 0; j < lastDim; j++) {
                        this.grad[offset + j] += result.data[offset + j] * (result.grad[offset + j] - dot);
                    }
                }
            }
        }, this);
        return result;
    }

    // --- ReLU ---

    public Tensor relu() {
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = Math.max(0, data[i]);
        }
        Tensor result = new Tensor(out, shape.clone(), !noGrad && requiresGrad);
        trackAutograd(result, () -> {
            if (this.requiresGrad) {
                this.ensureGrad();
                for (int i = 0; i < data.length; i++) {
                    this.grad[i] += (data[i] > 0 ? 1.0f : 0.0f) * result.grad[i];
                }
            }
        }, this);
        return result;
    }

    // --- Masked Fill ---

    /**
     * Replace elements where mask is true with the given value.
     * Mask is a 2D boolean array (T x T) applied per batch.
     * Input shape: (B, T, T).
     */
    public Tensor maskedFill(float[] mask, float value) {
        // mask has same size as data; 1.0 = keep, 0.0 = fill
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            int maskIdx = i % mask.length;
            out[i] = (mask[maskIdx] == 0.0f) ? value : data[i];
        }
        Tensor result = new Tensor(out, shape.clone(), !noGrad && requiresGrad);
        trackAutograd(result, () -> {
            if (this.requiresGrad) {
                this.ensureGrad();
                for (int i = 0; i < data.length; i++) {
                    int maskIdx = i % mask.length;
                    this.grad[i] += (mask[maskIdx] == 0.0f) ? 0.0f : result.grad[i];
                }
            }
        }, this);
        return result;
    }

    // --- Concatenation ---

    /**
     * Concatenate tensors along the last dimension.
     * All tensors must have the same shape except in the last dimension.
     */
    public static Tensor cat(List<Tensor> tensors, int dim) {
        if (dim < 0) dim += tensors.get(0).ndim();
        assert dim == tensors.get(0).ndim() - 1 : "cat currently only supports last dimension";

        int ndim = tensors.get(0).ndim();
        int outerSize = 1;
        for (int d = 0; d < ndim - 1; d++) {
            outerSize *= tensors.get(0).shape[d];
        }

        int totalLastDim = 0;
        int[] lastDims = new int[tensors.size()];
        for (int t = 0; t < tensors.size(); t++) {
            lastDims[t] = tensors.get(t).shape[ndim - 1];
            totalLastDim += lastDims[t];
        }

        int[] newShape = tensors.get(0).shape.clone();
        newShape[ndim - 1] = totalLastDim;

        boolean reqGrad = false;
        for (Tensor t : tensors) reqGrad |= t.requiresGrad;

        float[] out = new float[outerSize * totalLastDim];
        for (int i = 0; i < outerSize; i++) {
            int outOffset = i * totalLastDim;
            int catOffset = 0;
            for (int t = 0; t < tensors.size(); t++) {
                int ld = lastDims[t];
                System.arraycopy(tensors.get(t).data, i * ld, out, outOffset + catOffset, ld);
                catOffset += ld;
            }
        }

        Tensor result = new Tensor(out, newShape, !noGrad && reqGrad);
        if (!noGrad) {
            for (Tensor t : tensors) result.parents.add(t);
            final int outerSz = outerSize;
            final int totalLD = totalLastDim;
            result.backwardFn = () -> {
                int catOff = 0;
                for (int t = 0; t < tensors.size(); t++) {
                    Tensor tensor = tensors.get(t);
                    if (tensor.requiresGrad) {
                        tensor.ensureGrad();
                        int ld = lastDims[t];
                        for (int i = 0; i < outerSz; i++) {
                            for (int j = 0; j < ld; j++) {
                                tensor.grad[i * ld + j] += result.grad[i * totalLD + catOff + j];
                            }
                        }
                    }
                    catOff += lastDims[t];
                }
            };
        }
        return result;
    }

    // --- Select Along Dimension ---

    /**
     * Select a single index along a dimension, reducing that dimension.
     * E.g., selectAlongDim(1, -1) on (B, T, C) with index T-1 gives (B, C).
     */
    public Tensor selectAlongDim(int dim, int index) {
        if (dim < 0) dim += ndim();
        if (index < 0) index += shape[dim];

        int[] newShape = new int[ndim() - 1];
        int ni = 0;
        for (int d = 0; d < ndim(); d++) {
            if (d != dim) newShape[ni++] = shape[d];
        }

        final int dimSize = shape[dim];
        final int selIdx = index;
        int outerSz = 1;
        for (int d = 0; d < dim; d++) outerSz *= shape[d];
        int innerSz = 1;
        for (int d = dim + 1; d < ndim(); d++) innerSz *= shape[d];
        final int outerSize = outerSz;
        final int innerSize = innerSz;

        float[] out = new float[outerSize * innerSize];
        for (int o = 0; o < outerSize; o++) {
            int srcBase = o * dimSize * innerSize + selIdx * innerSize;
            System.arraycopy(data, srcBase, out, o * innerSize, innerSize);
        }

        Tensor result = new Tensor(out, newShape, !noGrad && requiresGrad);
        trackAutograd(result, () -> {
            if (this.requiresGrad) {
                this.ensureGrad();
                for (int o = 0; o < outerSize; o++) {
                    int srcBase = o * dimSize * innerSize + selIdx * innerSize;
                    for (int i = 0; i < innerSize; i++) {
                        this.grad[srcBase + i] += result.grad[o * innerSize + i];
                    }
                }
            }
        }, this);
        return result;
    }

    // --- Slice Along Dimension ---

    /**
     * Slice along a dimension: [start, end).
     * E.g., sliceAlongDim(1, start, end) on (B, T, C) gives (B, end-start, C).
     */
    public Tensor sliceAlongDim(int dim, int start, int end) {
        if (dim < 0) dim += ndim();
        if (start < 0) start += shape[dim];
        if (end < 0) end += shape[dim];
        final int sliceLen = end - start;

        int[] newShape = shape.clone();
        newShape[dim] = sliceLen;

        final int dimSize = shape[dim];
        final int slStart = start;
        int outerSz = 1;
        for (int d = 0; d < dim; d++) outerSz *= shape[d];
        int innerSz = 1;
        for (int d = dim + 1; d < ndim(); d++) innerSz *= shape[d];
        final int outerSize = outerSz;
        final int innerSize = innerSz;

        float[] out = new float[outerSize * sliceLen * innerSize];
        for (int o = 0; o < outerSize; o++) {
            for (int s = 0; s < sliceLen; s++) {
                int srcBase = o * dimSize * innerSize + (slStart + s) * innerSize;
                int dstBase = o * sliceLen * innerSize + s * innerSize;
                System.arraycopy(data, srcBase, out, dstBase, innerSize);
            }
        }

        Tensor result = new Tensor(out, newShape, !noGrad && requiresGrad);
        trackAutograd(result, () -> {
            if (this.requiresGrad) {
                this.ensureGrad();
                for (int o = 0; o < outerSize; o++) {
                    for (int s = 0; s < sliceLen; s++) {
                        int srcBase = o * dimSize * innerSize + (slStart + s) * innerSize;
                        int dstBase = o * sliceLen * innerSize + s * innerSize;
                        for (int i = 0; i < innerSize; i++) {
                            this.grad[srcBase + i] += result.grad[dstBase + i];
                        }
                    }
                }
            }
        }, this);
        return result;
    }

    // --- Embedding Lookup ---

    /**
     * Look up embeddings from a table.
     * this = embedding table of shape (vocabSize, embDim)
     * indices = flat int array of token indices
     * indicesShape = shape of indices (e.g., {B, T})
     * Returns tensor of shape (*indicesShape, embDim).
     */
    public Tensor embeddingLookup(int[] indices, int[] indicesShape) {
        int embDim = shape[1];
        int numIndices = indices.length;
        int[] outShape = new int[indicesShape.length + 1];
        System.arraycopy(indicesShape, 0, outShape, 0, indicesShape.length);
        outShape[outShape.length - 1] = embDim;

        float[] out = new float[numIndices * embDim];
        for (int i = 0; i < numIndices; i++) {
            int idx = indices[i];
            System.arraycopy(data, idx * embDim, out, i * embDim, embDim);
        }

        Tensor result = new Tensor(out, outShape, !noGrad && requiresGrad);
        trackAutograd(result, () -> {
            if (this.requiresGrad) {
                this.ensureGrad();
                for (int i = 0; i < numIndices; i++) {
                    int idx = indices[i];
                    for (int j = 0; j < embDim; j++) {
                        this.grad[idx * embDim + j] += result.grad[i * embDim + j];
                    }
                }
            }
        }, this);
        return result;
    }

    // --- Cross Entropy Loss ---

    /**
     * Cross entropy loss: combines softmax + negative log likelihood.
     * logits: (N, C), targets: int array of length N.
     * Returns scalar tensor (shape {1}).
     */
    public static Tensor crossEntropy(Tensor logits, int[] targets) {
        int N = logits.shape[0];
        int C = logits.shape[1];

        // Compute softmax
        float[] softmax = new float[N * C];
        for (int i = 0; i < N; i++) {
            int offset = i * C;
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < C; j++) max = Math.max(max, logits.data[offset + j]);
            float sumExp = 0;
            for (int j = 0; j < C; j++) {
                softmax[offset + j] = (float) Math.exp(logits.data[offset + j] - max);
                sumExp += softmax[offset + j];
            }
            for (int j = 0; j < C; j++) softmax[offset + j] /= sumExp;
        }

        // Compute loss: mean of -log(softmax[target])
        float loss = 0;
        for (int i = 0; i < N; i++) {
            loss += -(float) Math.log(softmax[i * C + targets[i]] + 1e-9f);
        }
        loss /= N;

        Tensor result = new Tensor(new float[]{loss}, new int[]{1}, !noGrad && logits.requiresGrad);
        trackAutograd(result, () -> {
            if (logits.requiresGrad) {
                logits.ensureGrad();
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < C; j++) {
                        float grad = softmax[i * C + j];
                        if (j == targets[i]) grad -= 1.0f;
                        logits.grad[i * C + j] += grad / N * result.grad[0];
                    }
                }
            }
        }, logits);
        return result;
    }

    // --- Layer Normalization ---

    /**
     * Layer normalization along the last dimension.
     * gamma and beta are learnable parameters of shape (lastDim,).
     */
    public static Tensor layerNorm(Tensor x, Tensor gamma, Tensor beta, float eps) {
        int lastDim = x.shape[x.ndim() - 1];
        int outerSize = x.numel() / lastDim;

        float[] out = new float[x.data.length];
        float[] means = new float[outerSize];
        float[] invStds = new float[outerSize];

        for (int i = 0; i < outerSize; i++) {
            int offset = i * lastDim;
            // Compute mean
            float mean = 0;
            for (int j = 0; j < lastDim; j++) mean += x.data[offset + j];
            mean /= lastDim;
            means[i] = mean;
            // Compute variance
            float var = 0;
            for (int j = 0; j < lastDim; j++) {
                float diff = x.data[offset + j] - mean;
                var += diff * diff;
            }
            var /= lastDim;
            float invStd = 1.0f / (float) Math.sqrt(var + eps);
            invStds[i] = invStd;
            // Normalize and scale
            for (int j = 0; j < lastDim; j++) {
                float normalized = (x.data[offset + j] - mean) * invStd;
                out[offset + j] = normalized * gamma.data[j] + beta.data[j];
            }
        }

        boolean reqGrad = !noGrad && (x.requiresGrad || gamma.requiresGrad || beta.requiresGrad);
        Tensor result = new Tensor(out, x.shape.clone(), reqGrad);
        if (noGrad) return result;
        result.parents.add(x);
        result.parents.add(gamma);
        result.parents.add(beta);
        result.backwardFn = () -> {
            for (int i = 0; i < outerSize; i++) {
                int offset = i * lastDim;
                float mean = means[i];
                float invStd = invStds[i];

                // Compute dgamma, dbeta
                if (gamma.requiresGrad) gamma.ensureGrad();
                if (beta.requiresGrad) beta.ensureGrad();

                float[] xhat = new float[lastDim];
                for (int j = 0; j < lastDim; j++) {
                    xhat[j] = (x.data[offset + j] - mean) * invStd;
                    if (gamma.requiresGrad) {
                        gamma.grad[j] += result.grad[offset + j] * xhat[j];
                    }
                    if (beta.requiresGrad) {
                        beta.grad[j] += result.grad[offset + j];
                    }
                }

                // Compute dx
                if (x.requiresGrad) {
                    x.ensureGrad();
                    // dxhat = dout * gamma
                    float[] dxhat = new float[lastDim];
                    for (int j = 0; j < lastDim; j++) {
                        dxhat[j] = result.grad[offset + j] * gamma.data[j];
                    }
                    // dx = invStd * (dxhat - mean(dxhat) - xhat * mean(dxhat * xhat))
                    float dxhatMean = 0;
                    float dxhatXhatMean = 0;
                    for (int j = 0; j < lastDim; j++) {
                        dxhatMean += dxhat[j];
                        dxhatXhatMean += dxhat[j] * xhat[j];
                    }
                    dxhatMean /= lastDim;
                    dxhatXhatMean /= lastDim;
                    for (int j = 0; j < lastDim; j++) {
                        x.grad[offset + j] += invStd * (dxhat[j] - dxhatMean - xhat[j] * dxhatXhatMean);
                    }
                }
            }
        };
        return result;
    }

    // --- Dropout ---

    /**
     * Dropout: randomly zero out elements with probability p during training.
     * mask: pre-generated boolean mask (true = keep). null if not training.
     */
    public Tensor dropout(boolean[] mask, float p) {
        if (mask == null) {
            // Inference mode: no dropout
            return this;
        }
        float scale = 1.0f / (1.0f - p);
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = mask[i] ? data[i] * scale : 0;
        }
        Tensor result = new Tensor(out, shape.clone(), !noGrad && requiresGrad);
        trackAutograd(result, () -> {
            if (this.requiresGrad) {
                this.ensureGrad();
                for (int i = 0; i < data.length; i++) {
                    this.grad[i] += mask[i] ? result.grad[i] * scale : 0;
                }
            }
        }, this);
        return result;
    }

    // ---- Utility Methods ----

    /** Sample from a probability distribution (1D tensor). Returns the index. */
    public int multinomial(Random rng) {
        float r = rng.nextFloat();
        float cumSum = 0;
        for (int i = 0; i < data.length; i++) {
            cumSum += data[i];
            if (r < cumSum) return i;
        }
        return data.length - 1;
    }

    /** Detach from computation graph (return a new tensor sharing data but no grad tracking). */
    public Tensor detach() {
        return new Tensor(data.clone(), shape.clone(), false);
    }

    /** Get a float item from a scalar (1-element) tensor. */
    public float item() {
        return data[0];
    }

    @Override
    public String toString() {
        return "Tensor(shape=" + Arrays.toString(shape) + ", data=" +
                (data.length <= 10 ? Arrays.toString(data) : "[" + data[0] + ", ..., " + data[data.length - 1] + "]") +
                ")";
    }
}
