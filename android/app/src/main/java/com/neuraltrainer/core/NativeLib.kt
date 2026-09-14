package com.neuraltrainer.core

/**
 * Native library interface for high-performance neural network operations
 */
object NativeLib {
    
    init {
        System.loadLibrary("neuraltrainer_core")
    }
    
    /**
     * Initialize the native library
     * @return 0 on success, error code otherwise
     */
    external fun initialize(): Int
    
    /**
     * Build a new neural network
     * @param inputs Number of input features
     * @param outputs Number of output neurons
     * @param hiddenLayers JSON array of layer sizes (e.g., "[64,32]")
     * @param activation Activation function name ("relu", "tanh", "sigmoid", etc.)
     * @param optimizer Optimizer name ("sgd", "adam", "momentum", etc.)
     * @param lr Learning rate
     * @param loss Loss function name ("mse", "bce", "mae")
     * @return Session ID or -1 on error
     */
    external fun buildNetwork(
        inputs: Int,
        outputs: Int,
        hiddenLayers: String,
        activation: String,
        optimizer: String,
        lr: Double,
        loss: String
    ): Int
    
    /**
     * Train the network for N steps
     * @param sessionId Session ID from buildNetwork
     * @param datasetJson JSON-encoded dataset
     * @param steps Number of training steps
     * @param lr Learning rate
     * @return Average loss
     */
    external fun trainSteps(
        sessionId: Int,
        datasetJson: String,
        steps: Int,
        lr: Double
    ): Double
    
    /**
     * Run prediction on input
     * @param sessionId Session ID
     * @param inputJson JSON-encoded input vector
     * @return JSON-encoded output vector
     */
    external fun predict(
        sessionId: Int,
        inputJson: String
    ): String
    
    /**
     * Get network snapshot for visualization
     * @param sessionId Session ID
     * @return JSON-encoded snapshot with weights, activations, topology
     */
    external fun getSnapshot(sessionId: Int): String
    
    /**
     * Export network to JSON
     * @param sessionId Session ID
     * @return JSON-encoded network with all weights
     */
    external fun exportNetwork(sessionId: Int): String
    
    /**
     * Import network from JSON
     * @param json JSON-encoded network
     * @return Session ID or -1 on error
     */
    external fun importNetwork(json: String): Int
    
    /**
     * Destroy a training session
     * @param sessionId Session ID to destroy
     */
    external fun destroySession(sessionId: Int)
    
    /**
     * Cleanup all sessions
     */
    external fun cleanup()
}
