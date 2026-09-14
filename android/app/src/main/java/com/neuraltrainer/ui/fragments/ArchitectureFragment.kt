package com.neuraltrainer.ui.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.neuraltrainer.R
import com.neuraltrainer.databinding.FragmentArchitectureBinding
import com.neuraltrainer.core.NeuralNetworkManager
import com.neuraltrainer.core.NetworkSnapshot

/**
 * Fragment displaying network architecture information
 */
class ArchitectureFragment : Fragment() {
    
    private var _binding: FragmentArchitectureBinding? = null
    private val binding get() = _binding!!
    
    var manager: NeuralNetworkManager? = null
    private var snapshot: NetworkSnapshot? = null
    
    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentArchitectureBinding.inflate(inflater, container, false)
        return binding.root
    }
    
    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        
        // Setup export/import buttons
        binding.exportButton.setOnClickListener {
            exportNetwork()
        }
        
        binding.importButton.setOnClickListener {
            importNetwork()
        }
        
        updateArchitecture()
    }
    
    /**
     * Update the architecture display with current network info
     */
    fun updateArchitecture() {
        val snapshot = manager?.getSnapshot()
        this.snapshot = snapshot
        
        if (snapshot == null) {
            binding.layerInfoText.text = "No network initialized"
            binding.paramCountText.text = "Total Parameters: 0"
            return
        }
        
        val sb = StringBuilder()
        sb.append("Input → ")
        for ((index, layer) in snapshot.layers.withIndex()) {
            sb.append("${layer.nOut} neurons")
            if (index < snapshot.layers.lastIndex) sb.append(" → ")
        }
        sb.append(" → Output")
        
        binding.layerInfoText.text = sb.toString()
        binding.paramCountText.text = "Total Parameters: ${calculateParameters(snapshot)}"
    }
    
    private fun calculateParameters(snapshot: NetworkSnapshot): Int {
        var total = 0
        for (layer in snapshot.layers) {
            // Weights + biases
            total += layer.nIn * layer.nOut + layer.nOut
        }
        return total
    }
    
    private fun exportNetwork() {
        val json = manager?.exportNetwork() ?: "{}"
        // In a real app, save to file or share
        android.widget.Toast.makeText(context, "Network exported (${json.length} chars)", android.widget.Toast.LENGTH_SHORT).show()
    }
    
    private fun importNetwork() {
        // In a real app, load from file
        android.widget.Toast.makeText(context, "Import not implemented", android.widget.Toast.LENGTH_SHORT).show()
    }
    
    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
