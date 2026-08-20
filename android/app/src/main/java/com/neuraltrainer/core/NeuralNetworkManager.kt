package com.neuraltrainer.core

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Kotlin wrapper around NativeLib for easier usage
 * Manages neural network sessions and provides high-level APIs
 */
class NeuralNetworkManager(private val context: Context) {
    
    private var sessionId = -1
    private val datasets = mutableMapOf<String, List<Pair<List<Double>, List<Double>>>>()
    
    init {
        NativeLib.initialize()
        generateDatasets()
    }
    
    /**
     * Build a new neural network with specified configuration
     */
    fun buildNetwork(
        inputs: Int,
        outputs: Int,
        hiddenLayers: List<Int>,
        activation: String = "relu",
        optimizer: String = "adam",
        learningRate: Double = 0.01,
        loss: String = "mse"
    ): Boolean {
        val hiddenLayersJson = hiddenLayers.joinToString(",", "[", "]")
        sessionId = NativeLib.buildNetwork(
            inputs = inputs,
            outputs = outputs,
            hiddenLayers = hiddenLayersJson,
            activation = activation,
            optimizer = optimizer,
            lr = learningRate,
            loss = loss
        )
        return sessionId > 0
    }
    
    /**
     * Train for specified number of steps
     * @return Average loss
     */
    fun trainSteps(steps: Int, learningRate: Double = 0.01): Double {
        if (sessionId < 0) return -1.0
        
        val datasetJson = getCurrentDatasetJson()
        return NativeLib.trainSteps(sessionId, datasetJson, steps, learningRate)
    }
    
    /**
     * Run prediction on input
     */
    fun predict(input: List<Double>): List<Double> {
        if (sessionId < 0) return emptyList()
        
        val inputJson = input.joinToString(",", "[", "]")
        val outputJson = NativeLib.predict(sessionId, inputJson)
        
        return parseDoubleArray(outputJson)
    }
    
    /**
     * Get network snapshot for visualization
     */
    fun getSnapshot(): NetworkSnapshot? {
        if (sessionId < 0) return null
        
        val jsonStr = NativeLib.getSnapshot(sessionId)
        return try {
            NetworkSnapshot.fromJson(jsonStr)
        } catch (e: Exception) {
            null
        }
    }
    
    /**
     * Export network to JSON string
     */
    fun exportNetwork(): String {
        return if (sessionId >= 0) {
            NativeLib.exportNetwork(sessionId)
        } else {
            "{}"
        }
    }
    
    /**
     * Import network from JSON string
     */
    fun importNetwork(json: String): Boolean {
        sessionId = NativeLib.importNetwork(json)
        return sessionId > 0
    }
    
    /**
     * Destroy current session
     */
    fun destroySession() {
        if (sessionId > 0) {
            NativeLib.destroySession(sessionId)
            sessionId = -1
        }
    }
    
    /**
     * Cleanup all sessions
     */
    fun cleanup() {
        destroySession()
        NativeLib.cleanup()
    }
    
    /**
     * Get current epoch count
     */
    fun getEpoch(): Int {
        return getSnapshot()?.epoch ?: 0
    }
    
    /**
     * Get loss history
     */
    fun getLossHistory(): List<Double> {
        return getSnapshot()?.lossHistory ?: emptyList()
    }
    
    /**
     * Generate training datasets for various tasks
     */
    private fun generateDatasets() {
        // XOR dataset
        datasets["xor"] = listOf(
            listOf(0.0, 0.0) to listOf(0.0),
            listOf(0.0, 1.0) to listOf(1.0),
            listOf(1.0, 0.0) to listOf(1.0),
            listOf(1.0, 1.0) to listOf(0.0)
        )
        
        // AND gate
        datasets["and"] = listOf(
            listOf(0.0, 0.0) to listOf(0.0),
            listOf(0.0, 1.0) to listOf(0.0),
            listOf(1.0, 0.0) to listOf(0.0),
            listOf(1.0, 1.0) to listOf(1.0)
        )
        
        // OR gate
        datasets["or"] = listOf(
            listOf(0.0, 0.0) to listOf(0.0),
            listOf(0.0, 1.0) to listOf(1.0),
            listOf(1.0, 0.0) to listOf(1.0),
            listOf(1.0, 1.0) to listOf(1.0)
        )
        
        // XNOR gate
        datasets["xnor"] = listOf(
            listOf(0.0, 0.0) to listOf(1.0),
            listOf(0.0, 1.0) to listOf(0.0),
            listOf(1.0, 0.0) to listOf(0.0),
            listOf(1.0, 1.0) to listOf(1.0)
        )
        
        // Parity (4-bit)
        datasets["parity"] = generateParityDataset()
        
        // Sine wave
        datasets["sine"] = generateSineDataset()
        
        // Circle classification
        datasets["circle"] = generateCircleDataset()
        
        // Spiral classification
        datasets["spiral"] = generateSpiralDataset()
        
        // 7-segment display
        datasets["seven_segment"] = generateSevenSegmentDataset()
        
        // Half adder
        datasets["half_adder"] = listOf(
            listOf(0.0, 0.0) to listOf(0.0, 0.0),
            listOf(0.0, 1.0) to listOf(1.0, 0.0),
            listOf(1.0, 0.0) to listOf(1.0, 0.0),
            listOf(1.0, 1.0) to listOf(0.0, 1.0)
        )
        
        // Autoencoder (identity)
        datasets["autoencoder"] = generateAutoencoderDataset()
    }
    
    private fun generateParityDataset(): List<Pair<List<Double>, List<Double>>> {
        val dataset = mutableListOf<Pair<List<Double>, List<Double>>>()
        for (i in 0 until 16) {
            val bits = listOf(
                ((i shr 3) and 1).toDouble(),
                ((i shr 2) and 1).toDouble(),
                ((i shr 1) and 1).toDouble(),
                (i and 1).toDouble()
            )
            val parity = bits.sum() % 2
            dataset.add(bits to listOf(parity.toDouble()))
        }
        return dataset
    }
    
    private fun generateSineDataset(): List<Pair<List<Double>, List<Double>>> {
        val dataset = mutableListOf<Pair<List<Double>, List<Double>>>()
        for (i in 0 until 50) {
            val x = i * 2.0 * Math.PI / 50
            dataset.add(listOf(x / (2 * Math.PI)) to listOf((Math.sin(x) + 1) / 2))
        }
        return dataset
    }
    
    private fun generateCircleDataset(): List<Pair<List<Double>, List<Double>>> {
        val dataset = mutableListOf<Pair<List<Double>, List<Double>>>()
        val radius = 0.5
        
        // Points inside circle
        for (i in 0 until 50) {
            val angle = i * 2 * Math.PI / 50
            val r = radius * 0.8 * Math.random()
            val x = r * Math.cos(angle)
            val y = r * Math.sin(angle)
            dataset.add(listOf(x, y) to listOf(1.0))
        }
        
        // Points outside circle
        for (i in 0 until 50) {
            val angle = i * 2 * Math.PI / 50
            val r = radius * (1.2 + 0.8 * Math.random())
            val x = r * Math.cos(angle)
            val y = r * Math.sin(angle)
            dataset.add(listOf(x, y) to listOf(0.0))
        }
        
        return dataset
    }
    
    private fun generateSpiralDataset(): List<Pair<List<Double>, List<Double>>> {
        val dataset = mutableListOf<Pair<List<Double>, List<Double>>>()
        val n = 50
        
        // Class 0 spiral
        for (i in 0 until n) {
            val t = i * 4 * Math.PI / n
            val r = 0.1 + 0.5 * t / (4 * Math.PI)
            val x = r * Math.cos(t)
            val y = r * Math.sin(t)
            dataset.add(listOf(x, y) to listOf(0.0))
        }
        
        // Class 1 spiral (rotated)
        for (i in 0 until n) {
            val t = i * 4 * Math.PI / n + Math.PI
            val r = 0.1 + 0.5 * t / (4 * Math.PI)
            val x = r * Math.cos(t)
            val y = r * Math.sin(t)
            dataset.add(listOf(x, y) to listOf(1.0))
        }
        
        return dataset
    }
    
    private fun generateSevenSegmentDataset(): List<Pair<List<Double>, List<Double>>> {
        // 7-segment encoding for digits 0-9
        val segments = arrayOf(
            doubleArrayOf(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0), // 0
            doubleArrayOf(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0), // 1
            doubleArrayOf(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0), // 2
            doubleArrayOf(1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0), // 3
            doubleArrayOf(0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0), // 4
            doubleArrayOf(1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0), // 5
            doubleArrayOf(1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0), // 6
            doubleArrayOf(1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0), // 7
            doubleArrayOf(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0), // 8
            doubleArrayOf(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0)  // 9
        )
        
        val dataset = mutableListOf<Pair<List<Double>, List<Double>>>()
        for (digit in 0..9) {
            // Create one-hot encoded output
            val output = List(7) { 0.0 }.toMutableList()
            // For simplicity, we'll use digit recognition (output = digit index normalized)
            output[digit % 7] = 1.0
            dataset.add(segments[digit].toList() to output)
        }
        return dataset
    }
    
    private fun generateAutoencoderDataset(): List<Pair<List<Double>, List<Double>>> {
        val dataset = mutableListOf<Pair<List<Double>, List<Double>>>()
        // Generate random 8-bit patterns
        for (i in 0 until 256) {
            val pattern = List(8) { j -> ((i shr j) and 1).toDouble() }
            dataset.add(pattern to pattern)
        }
        return dataset
    }
    
    /**
     * Get dataset JSON for current task
     */
    private fun getCurrentDatasetJson(): String {
        // Default to XOR if no specific task selected
        val dataset = datasets["xor"] ?: return "[]"
        
        val jsonArray = JSONArray()
        for ((input, output) in dataset) {
            val obj = JSONObject()
            obj.put("x", JSONArray(input))
            obj.put("y", JSONArray(output))
            jsonArray.put(obj)
        }
        
        return jsonArray.toString()
    }
    
    /**
     * Parse JSON array of doubles
     */
    private fun parseDoubleArray(json: String): List<Double> {
        return try {
            val arr = JSONArray(json)
            List(arr.length()) { arr.getDouble(it) }
        } catch (e: Exception) {
            emptyList()
        }
    }
}

/**
 * Data class for network snapshot
 */
data class NetworkSnapshot(
    val epoch: Int,
    val optimizer: String,
    val learningRate: Double,
    val loss: String,
    val layers: List<LayerInfo>,
    val lossHistory: List<Double>
) {
    data class LayerInfo(
        val type: String,
        val nIn: Int,
        val nOut: Int,
        val activation: String
    )
    
    companion object {
        fun fromJson(json: String): NetworkSnapshot {
            val obj = JSONObject(json)
            val epoch = obj.optInt("epoch", 0)
            val optimizer = obj.optString("optimizer", "unknown")
            val learningRate = obj.optDouble("learning_rate", 0.01)
            val loss = obj.optString("loss", "mse")
            
            val layersJson = obj.optJSONArray("layers") ?: JSONArray()
            val layers = List(layersJson.length()) { i ->
                val layer = layersJson.getJSONObject(i)
                LayerInfo(
                    type = layer.optString("type", "dense"),
                    nIn = layer.optInt("n_in", 0),
                    nOut = layer.optInt("n_out", 0),
                    activation = layer.optString("activation", "relu")
                )
            }
            
            val lossHistoryJson = obj.optJSONArray("loss_history") ?: JSONArray()
            val lossHistory = List(lossHistoryJson.length()) { i ->
                lossHistoryJson.getDouble(i)
            }
            
            return NetworkSnapshot(epoch, optimizer, learningRate, loss, layers, lossHistory)
        }
    }
}
