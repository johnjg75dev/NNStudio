package com.neuraltrainer.ui.fragments

import android.graphics.Color
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.XAxis
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import com.neuraltrainer.R
import com.neuraltrainer.core.NeuralNetworkManager
import kotlinx.coroutines.*

/**
 * Fragment displaying loss chart during training
 */
class LossChartFragment : Fragment() {
    
    private var _binding: View? = null
    private lateinit var lossChart: LineChart
    
    var manager: NeuralNetworkManager? = null
    private var chartUpdateJob: Job? = null
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = inflater.inflate(R.layout.fragment_loss_chart, container, false)
        return _binding!!
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        lossChart = view.findViewById(R.id.lossChart)
        setupChart()
        
        // Start chart update loop
        startChartUpdateLoop()
    }
    
    private fun setupChart() {
        lossChart.apply {
            description.isEnabled = false
            setTouchEnabled(true)
            isDragEnabled = true
            setScaleEnabled(true)
            setPinchZoom(true)
            
            xAxis.position = XAxis.XAxisPosition.BOTTOM
            xAxis.textColor = Color.WHITE
            xAxis.setDrawGridLines(false)
            
            axisLeft.textColor = Color.WHITE
            axisRight.isEnabled = false
            
            legend.textColor = Color.WHITE
            
            setBackgroundColor(Color.parseColor("#1A1A2E"))
        }
    }
    
    private fun startChartUpdateLoop() {
        chartUpdateJob = lifecycleScope.launch {
            while (isActive) {
                val lossHistory = manager?.getLossHistory() ?: emptyList()
                
                if (lossHistory.isNotEmpty()) {
                    withContext(Dispatchers.Main) {
                        updateChart(lossHistory)
                    }
                }
                delay(200) // Update every 200ms
            }
        }
    }
    
    private fun updateChart(lossHistory: List<Double>) {
        val entries = lossHistory.mapIndexed { index, value ->
            Entry(index.toFloat(), value.toFloat())
        }
        
        val dataSet = LineDataSet(entries, "Loss").apply {
            color = Color.parseColor("#FF5722")
            lineWidth = 2f
            setDrawCircles(false)
            setDrawValues(false)
            mode = LineDataSet.Mode.LINEAR
        }
        
        val data = LineData(dataSet)
        lossChart.data = data
        lossChart.invalidate()
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        chartUpdateJob?.cancel()
        _binding = null
    }
}
