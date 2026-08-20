package com.neuraltrainer.ui.fragments

import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.neuraltrainer.R
import com.neuraltrainer.core.NeuralNetworkManager
import com.neuraltrainer.core.NetworkSnapshot
import kotlinx.coroutines.*

/**
 * Fragment displaying network architecture visualization
 */
class NetworkViewFragment : Fragment() {
    
    private var _binding: View? = null
    private lateinit var networkCanvas: NetworkCanvasView
    
    var manager: NeuralNetworkManager? = null
    private var visualizationJob: Job? = null
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = inflater.inflate(R.layout.fragment_network_view, container, false)
        return _binding!!
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        networkCanvas = view.findViewById(R.id.networkCanvas)
        
        // Start visualization loop
        startVisualizationLoop()
    }
    
    private fun startVisualizationLoop() {
        visualizationJob = lifecycleScope.launch {
            while (isActive) {
                val snapshot = manager?.getSnapshot()
                
                if (snapshot != null) {
                    withContext(Dispatchers.Main) {
                        networkCanvas.setSnapshot(snapshot)
                        networkCanvas.invalidate()
                    }
                }
                delay(50) // 20 FPS
            }
        }
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        visualizationJob?.cancel()
        _binding = null
    }
}

/**
 * Custom view for drawing neural network architecture
 */
class NetworkCanvasView @JvmOverloads constructor(
    context: android.content.Context,
    attrs: android.util.AttributeSet? = null
) : View(context, attrs) {
    
    private var snapshot: NetworkSnapshot? = null
    
    private val nodePaint = Paint().apply {
        color = Color.parseColor("#3F80FF")
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    
    private val connectionPaint = Paint().apply {
        color = Color.parseColor("#40808080")
        strokeWidth = 2f
        isAntiAlias = true
    }
    
    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 12f
        textAlign = Paint.Align.CENTER
        isAntiAlias = true
    }
    
    fun setSnapshot(snap: NetworkSnapshot?) {
        snapshot = snap
    }
    
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        
        val snap = snapshot ?: return
        
        canvas.drawColor(Color.parseColor("#1A1A2E"))
        
        val layers = snap.layers
        if (layers.isEmpty()) return
        
        val width = width.toFloat()
        val height = height.toFloat()
        
        val layerSpacing = width / (layers.size + 1)
        val maxNeurons = layers.maxOf { it.nOut }.coerceAtLeast(1)
        val neuronSpacing = height / (maxNeurons + 1)
        
        // Store neuron positions for drawing connections
        val neuronPositions = mutableListOf<List<Pair<Float, Float>>>()
        
        // Draw neurons
        for ((layerIdx, layer) in layers.withIndex()) {
            val x = layerSpacing * (layerIdx + 1)
            val neuronCount = layer.nOut.coerceAtMost(maxNeurons)
            val layerPositions = mutableListOf<Pair<Float, Float>>()
            
            for (neuronIdx in 0 until neuronCount) {
                val y = neuronSpacing * (neuronIdx + 1)
                layerPositions.add(x to y)
                
                // Draw connection lines to previous layer
                if (layerIdx > 0) {
                    val prevLayer = neuronPositions[layerIdx - 1]
                    for ((prevX, prevY) in prevLayer) {
                        // Color based on weight (placeholder - would need actual weights)
                        canvas.drawLine(prevX, prevY, x, y, connectionPaint)
                    }
                }
                
                // Draw neuron circle
                canvas.drawCircle(x, y, 15f, nodePaint)
                
                // Draw activation label for small networks
                if (layers.size <= 4 && neuronCount <= 8) {
                    canvas.drawText(layer.activation, x, y + 30, textPaint)
                }
            }
            
            neuronPositions.add(layerPositions)
        }
        
        // Draw layer labels
        textPaint.textSize = 14f
        for ((layerIdx, layer) in layers.withIndex()) {
            val x = layerSpacing * (layerIdx + 1)
            canvas.drawText("L${layerIdx + 1}", x, 25f, textPaint)
        }
    }
}
