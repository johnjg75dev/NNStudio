package com.neuraltrainer.ui.activities

import android.os.Bundle
import android.view.View
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.button.MaterialButton
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator
import com.google.android.material.textfield.TextInputEditText
import com.neuraltrainer.R
import com.neuraltrainer.core.NativeLib
import com.neuraltrainer.core.NeuralNetworkManager
import com.neuraltrainer.ui.adapters.MainPagerAdapter
import com.neuraltrainer.ui.fragments.ArchitectureFragment
import com.neuraltrainer.ui.fragments.LossChartFragment
import com.neuraltrainer.ui.fragments.NetworkViewFragment
import com.neuraltrainer.ui.fragments.TestFragment
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * Main Activity - Neural Network Trainer
 * Provides UI for building, training, and visualizing neural networks
 */
class MainActivity : AppCompatActivity() {
    
    // Native library session ID
    private var sessionId = -1
    
    // Network manager (Kotlin wrapper around native lib)
    private lateinit var networkManager: NeuralNetworkManager
    
    // Training state
    private var isTraining = false
    private var trainingJob: Job? = null
    private var epoch = 0
    
    // UI Views
    private lateinit var networkView: NetworkCanvasView
    private lateinit var trainPauseButton: MaterialButton
    private lateinit var buildButton: MaterialButton
    private lateinit var resetButton: MaterialButton
    
    // Fragment references
    private var networkViewFragment: NetworkViewFragment? = null
    private var lossChartFragment: LossChartFragment? = null
    private var testFragment: TestFragment? = null
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        // Initialize native library
        val initResult = NativeLib.initialize()
        if (initResult != 0) {
            showError("Failed to initialize native library")
        }
        
        // Initialize network manager
        networkManager = NeuralNetworkManager(this)
        
        // Setup UI
        setupViews()
        setupSpinners()
        setupTabs()
        setupListeners()
    }
    
    private fun setupViews() {
        trainPauseButton = findViewById(R.id.trainPauseButton)
        buildButton = findViewById(R.id.buildButton)
        resetButton = findViewById(R.id.resetButton)
    }
    
    private fun setupSpinners() {
        // Setup activation function spinner
        val activationSpinner: android.widget.Spinner = findViewById(R.id.activationSpinner)
        val activations = arrayOf("relu", "leaky_relu", "tanh", "sigmoid", "gelu", "swish")
        activationSpinner.adapter = android.widget.ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            activations.map { it.uppercase() }.toTypedArray()
        )
        
        // Setup optimizer spinner
        val optimizerSpinner: android.widget.Spinner = findViewById(R.id.optimizerSpinner)
        val optimizers = arrayOf("sgd", "momentum", "rmsprop", "adam", "adamw")
        optimizerSpinner.adapter = android.widget.ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            optimizers.map { it.uppercase() }.toTypedArray()
        )
        
        // Setup loss function spinner
        val lossSpinner: android.widget.Spinner = findViewById(R.id.lossSpinner)
        val losses = arrayOf("mse", "bce", "mae")
        lossSpinner.adapter = android.widget.ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            losses.map { it.uppercase() }.toTypedArray()
        )
        
        // Setup task spinner
        val taskSpinner: android.widget.Spinner = findViewById(R.id.taskSpinner)
        val tasks = arrayOf(
            "XOR Gate", "AND Gate", "OR Gate", "XNOR Gate",
            "7-Segment", "Parity", "Sine", "Half Adder",
            "Circle", "Spiral", "Autoencoder"
        )
        taskSpinner.adapter = android.widget.ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            tasks
        )
        
        // Setup architecture spinner
        val archSpinner: android.widget.Spinner = findViewById(R.id.archSpinner)
        val architectures = arrayOf("MLP", "CNN", "RNN", "Transformer")
        archSpinner.adapter = android.widget.ArrayAdapter(
            this,
            android.R.layout.simple_spinner_item,
            architectures
        )
    }
    
    private fun setupTabs() {
        val tabLayout: TabLayout = findViewById(R.id.tabLayout)
        val viewPager: ViewPager2 = findViewById(R.id.viewPager)
        
        // Setup view pager with tabs
        viewPager.adapter = MainPagerAdapter(this)
        
        TabLayoutMediator(tabLayout, viewPager) { tab, position ->
            tab.text = when (position) {
                0 -> "Network"
                1 -> "Loss Chart"
                2 -> "Test"
                3 -> "Architecture"
                else -> "Tab"
            }
        }.attach()
    }
    
    private fun setupListeners() {
        buildButton.setOnClickListener {
            buildNetwork()
        }
        
        trainPauseButton.setOnClickListener {
            toggleTraining()
        }
        
        resetButton.setOnClickListener {
            resetNetwork()
        }
    }
    
    private fun buildNetwork() {
        try {
            // Get configuration from UI
            val hiddenLayersStr = findViewById<com.google.android.material.textfield.TextInputEditText>(R.id.hiddenLayersEdit).text.toString()
            val neuronsStr = findViewById<com.google.android.material.textfield.TextInputEditText>(R.id.neuronsEdit).text.toString()
            val lrStr = findViewById<com.google.android.material.textfield.TextInputEditText>(R.id.learningRateEdit).text.toString()
            
            val hiddenLayers = hiddenLayersStr.toIntOrNull() ?: 2
            val neurons = neuronsStr.toIntOrNull() ?: 16
            val learningRate = lrStr.toDoubleOrNull() ?: 0.01
            
            // Build hidden layers array string
            val hiddenLayersJson = buildHiddenLayersJson(hiddenLayers, neurons)
            
            // Get selected options
            val activationSpinner: android.widget.Spinner = findViewById(R.id.activationSpinner)
            val optimizerSpinner: android.widget.Spinner = findViewById(R.id.optimizerSpinner)
            val lossSpinner: android.widget.Spinner = findViewById(R.id.lossSpinner)
            
            val activation = activationSpinner.selectedItem.toString().lowercase()
            val optimizer = optimizerSpinner.selectedItem.toString().lowercase()
            val loss = lossSpinner.selectedItem.toString().lowercase()
            
            // Determine inputs/outputs based on task
            val taskSpinner: android.widget.Spinner = findViewById(R.id.taskSpinner)
            val (inputs, outputs) = getTaskDimensions(taskSpinner.selectedItemPosition)
            
            // Call native library to build network
            sessionId = NativeLib.buildNetwork(
                inputs = inputs,
                outputs = outputs,
                hiddenLayers = hiddenLayersJson,
                activation = activation,
                optimizer = optimizer,
                lr = learningRate,
                loss = loss
            )
            
            if (sessionId > 0) {
                showSuccess("Network built successfully")
                updateStats()
                
                // Start visualization loop
                startVisualizationLoop()
            } else {
                showError("Failed to build network")
            }
            
        } catch (e: Exception) {
            showError("Error: ${e.message}")
        }
    }
    
    private fun buildHiddenLayersJson(count: Int, neurons: Int): String {
        return List(count) { neurons }.joinToString(",", "[", "]")
    }
    
    private fun getTaskDimensions(taskIndex: Int): Pair<Int, Int> {
        return when (taskIndex) {
            0, 1, 2, 3 -> Pair(2, 1)  // Logic gates
            4 -> Pair(4, 7)           // 7-segment
            5 -> Pair(4, 1)           // Parity
            6 -> Pair(1, 1)           // Sine
            7 -> Pair(2, 2)           // Half adder
            8, 9 -> Pair(2, 1)        // Circle, Spiral
            10 -> Pair(8, 8)          // Autoencoder
            else -> Pair(2, 1)
        }
    }
    
    private fun toggleTraining() {
        if (isTraining) {
            pauseTraining()
        } else {
            startTraining()
        }
    }
    
    private fun startTraining() {
        if (sessionId < 0) {
            showError("Build a network first")
            return
        }
        
        isTraining = true
        trainPauseButton.text = "Pause"
        
        trainingJob = lifecycleScope.launch {
            while (isActive && isTraining) {
                // Train for N steps
                val datasetJson = getCurrentDatasetJson()
                val loss = withContext(Dispatchers.Default) {
                    NativeLib.trainSteps(sessionId, datasetJson, 10, 0.01)
                }
                
                epoch += 10
                updateStats(loss)
                
                delay(100) // Update every 100ms
            }
        }
    }
    
    private fun pauseTraining() {
        isTraining = false
        trainPauseButton.text = "Train"
        trainingJob?.cancel()
    }
    
    private fun resetNetwork() {
        pauseTraining()
        epoch = 0
        
        if (sessionId > 0) {
            NativeLib.destroySession(sessionId)
            sessionId = -1
        }
        
        updateStats()
        showSuccess("Network reset")
    }
    
    private fun updateStats(loss: Double = 0.0) {
        withContext(Dispatchers.Main) {
            findViewById<android.widget.TextView>(R.id.epochText).text = "Epoch: $epoch"
            findViewById<android.widget.TextView>(R.id.lossText).text = "Loss: %.4f".format(loss)
        }
    }
    
    private fun startVisualizationLoop() {
        lifecycleScope.launch {
            while (isActive) {
                if (sessionId > 0) {
                    val snapshot = withContext(Dispatchers.Default) {
                        NativeLib.getSnapshot(sessionId)
                    }
                    // Update network visualization with snapshot
                }
                delay(50) // 20 FPS
            }
        }
    }
    
    private fun getCurrentDatasetJson(): String {
        // Generate or retrieve dataset based on selected task
        // For now, return XOR dataset as example
        return """[{"x":[0,0],"y":[0]},{"x":[0,1],"y":[1]},{"x":[1,0],"y":[1]},{"x":[1,1],"y":[0]}]"""
    }
    
    private fun showSuccess(message: String) {
        // Show snackbar or toast
        android.widget.Toast.makeText(this, message, android.widget.Toast.LENGTH_SHORT).show()
    }
    
    private fun showError(message: String) {
        android.widget.Toast.makeText(this, message, android.widget.Toast.LENGTH_LONG).show()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        pauseTraining()
        if (sessionId > 0) {
            NativeLib.destroySession(sessionId)
        }
        NativeLib.cleanup()
    }
}
