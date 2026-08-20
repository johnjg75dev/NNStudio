package com.neuraltrainer.ui.adapters

import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentActivity
import androidx.viewpager2.adapter.FragmentStateAdapter
import com.neuraltrainer.core.NeuralNetworkManager
import com.neuraltrainer.ui.fragments.ArchitectureFragment
import com.neuraltrainer.ui.fragments.LossChartFragment
import com.neuraltrainer.ui.fragments.NetworkViewFragment
import com.neuraltrainer.ui.fragments.TestFragment

/**
 * ViewPager2 adapter for main activity tabs
 */
class MainPagerAdapter(
    fragmentActivity: FragmentActivity,
    private val networkManager: NeuralNetworkManager
) : FragmentStateAdapter(fragmentActivity) {
    
    private val fragmentCreators = listOf(
        { NetworkViewFragment() },
        { LossChartFragment() },
        { TestFragment() },
        { ArchitectureFragment() }
    )
    
    override fun getItemCount(): Int = fragmentCreators.size
    
    override fun createFragment(position: Int): Fragment {
        val fragment = fragmentCreators[position]()
        
        // Pass manager to fragments that need it
        when (fragment) {
            is NetworkViewFragment -> fragment.manager = networkManager
            is LossChartFragment -> fragment.manager = networkManager
            is TestFragment -> fragment.manager = networkManager
            is ArchitectureFragment -> fragment.manager = networkManager
        }
        
        return fragment
    }
}
