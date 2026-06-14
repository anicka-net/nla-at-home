/**
 * demo/interactive.js
 * 
 * Client-side module for interactive Gemma 3 1B inference using Transformers.js v3.
 * Extract activations from all layers and performs nearest-neighbor lookup
 * in PCA-projected activation space.
 */

import { AutoModel, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0/dist/transformers.min.js';

let model = null;
let tokenizer = null;
const MODEL_ID = 'onnx-community/gemma-3-1b-it'; // Assume this exists in 2026

/**
 * Load model and tokenizer.
 * @param {Function} onProgress - Callback for loading progress.
 */
export async function loadModel(onProgress) {
    console.log(`Loading model ${MODEL_ID}...`);
    
    // Check for WebGPU support
    const device = 'gpu' in navigator ? 'webgpu' : 'wasm';
    console.log(`Using device: ${device}`);

    tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
    
    model = await AutoModel.from_pretrained(MODEL_ID, {
        progress_callback: onProgress,
        device: device,
        dtype: 'q4', // 4-bit quantization for browser efficiency
    });

    console.log("Model loaded successfully.");
}

/**
 * Run inference on a prompt and extract activations for all layers.
 * @param {string} promptText - The user prompt.
 * @returns {Promise<Object>} - Object containing layer activations.
 */
export async function getActivations(promptText) {
    if (!model || !tokenizer) {
        throw new Error("Model not loaded. Call loadModel() first.");
    }

    // Apply chat template if applicable, or just tokenize
    // Gemma 3 1B IT usually expects a chat format
    const messages = [{ role: 'user', content: promptText }];
    const chat = tokenizer.apply_chat_template(messages, { tokenize: false, add_generation_prompt: true });
    const inputs = await tokenizer(chat);

    console.log("Running forward pass...");
    
    // We only need the activations at the last token position
    // output_hidden_states: true returns activations for all layers
    const outputs = await model(inputs, { output_hidden_states: true });
    
    // hidden_states is an array of tensors: [embeddings, layer0, layer1, ..., layer25]
    const hiddenStates = outputs.hidden_states;
    const seqLen = inputs.input_ids.dims[1];
    const lastTokenIdx = seqLen - 1;

    const result = {
        layers: []
    };

    // hiddenStates[0] is embeddings. We want layer outputs: index 1 to 26.
    for (let l = 0; l < 26; l++) {
        const layerState = hiddenStates[l + 1];
        // Extract vector at last token position: [batch, seq, hidden] -> [1, seq, 1152]
        // We want [lastTokenIdx, :]
        const data = layerState.data; // Float32Array
        const hiddenSize = layerState.dims[2];
        const offset = lastTokenIdx * hiddenSize;
        
        const activation = data.slice(offset, offset + hiddenSize);
        
        result.layers.push({
            layer: l,
            activation: activation
        });
    }

    return result;
}

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a, b) {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    const mag = Math.sqrt(normA) * Math.sqrt(normB);
    return mag === 0 ? 0 : dot / mag;
}

/**
 * Project a high-dimensional activation into PCA space.
 * @param {Float32Array} activation - 1152-dim vector.
 * @param {Array<Array<number>>} matrix - 256x1152 PCA components.
 */
function project(activation, matrix) {
    const projected = new Float32Array(matrix.length);
    for (let i = 0; i < matrix.length; i++) {
        let sum = 0;
        const row = matrix[i];
        for (let j = 0; j < activation.length; j++) {
            sum += row[j] * activation[j];
        }
        projected[i] = sum;
    }
    return projected;
}

/**
 * Given activations and a reference bank, find the nearest descriptions.
 * @param {Object} activationResult - Output from getActivations.
 * @param {Object} referenceBank - The PCA matrices and reference vectors.
 */
export function lookupDescriptions(activationResult, referenceBank) {
    const results = [];
    
    for (const layerData of activationResult.layers) {
        const l = layerData.layer;
        const activation = layerData.activation;
        
        const pcaMatrix = referenceBank.pca_matrices[l];
        const referenceVectors = referenceBank.reference_vectors[l];
        const descriptions = referenceBank.descriptions[l];
        
        if (!pcaMatrix || !referenceVectors || !descriptions) {
            results.push({ layer: l, description: "Layer data missing in reference bank", similarity: 0 });
            continue;
        }

        // 1. Project to PCA space (1152 -> 256)
        const projected = project(activation, pcaMatrix);
        
        // 2. Find nearest neighbor in projected space
        let bestSim = -1;
        let bestIdx = -1;
        
        for (let i = 0; i < referenceVectors.length; i++) {
            const sim = cosineSimilarity(projected, referenceVectors[i]);
            if (sim > bestSim) {
                bestSim = sim;
                bestIdx = i;
            }
        }
        
        results.push({
            layer: l,
            description: descriptions[bestIdx],
            similarity: bestSim
        });
    }
    
    return results;
}
