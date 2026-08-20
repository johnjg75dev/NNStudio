package com.neuraltrainer.ui.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import com.neuraltrainer.R
import com.neuraltrainer.core.NeuralNetworkManager
import kotlinx.coroutines.*

/**
 * Fragment for testing trained network with custom inputs
 */
class TestFragment : Fragment() {
    
    private var _binding: View? = null
    
    var manager: NeuralNetworkManager? = null
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = inflater.inflate(R.layout.fragment_test, container, false)
        return _binding!!
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        val inputEdit = view.findViewById<TextInputEditText>(R.id.testInputEdit)
        val predictButton = view.findViewById<MaterialButton>(R.id.predictButton)
        val resultText = view.findViewById<android.widget.TextView>(R.id.resultText)
        
        predictButton.setOnClickListener {
            val manager = manager
            if (manager == null || !manager.isInitialized()) {
                Toast.makeText(context, "Build and train a network first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            
            val inputStr = inputEdit.text.toString()
            try {
                val input = inputStr.split(",").map { it.trim().toFloat() }.toFloatArray()
                
                lifecycleScope.launch(Dispatchers.Default) {
                    val output = manager.predictSingle(input)
                    
                    withContext(Dispatchers.Main) {
                        // Display result
                        resultText.text = "Output: ${output.joinToString(", ") { "%.4f".format(it) }}"
                    }
                }
            } catch (e: Exception) {
                Toast.makeText(context, "Invalid input format. Use comma-separated numbers.", Toast.LENGTH_SHORT).show()
            }
        }
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
