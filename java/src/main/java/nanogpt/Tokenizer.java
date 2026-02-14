package nanogpt;

import java.util.*;

/**
 * Character-level tokenizer.
 * Maps individual characters to integer indices and back.
 * Mirrors the encode/decode functions from the Python implementation.
 */
public class Tokenizer {

    private final Map<Character, Integer> stoi; // string to index
    private final Map<Integer, Character> itos; // index to string
    private final int vocabSize;

    public Tokenizer(String text) {
        // Find all unique characters and sort them
        Set<Character> charSet = new TreeSet<>();
        for (char c : text.toCharArray()) {
            charSet.add(c);
        }
        List<Character> chars = new ArrayList<>(charSet);

        this.vocabSize = chars.size();
        this.stoi = new HashMap<>();
        this.itos = new HashMap<>();

        for (int i = 0; i < chars.size(); i++) {
            stoi.put(chars.get(i), i);
            itos.put(i, chars.get(i));
        }
    }

    /** Encode a string into a sequence of integer indices. */
    public int[] encode(String s) {
        int[] result = new int[s.length()];
        for (int i = 0; i < s.length(); i++) {
            result[i] = stoi.get(s.charAt(i));
        }
        return result;
    }

    /** Decode a sequence of integer indices back into a string. */
    public String decode(int[] indices) {
        StringBuilder sb = new StringBuilder();
        for (int idx : indices) {
            sb.append(itos.get(idx));
        }
        return sb.toString();
    }

    public int getVocabSize() {
        return vocabSize;
    }
}
