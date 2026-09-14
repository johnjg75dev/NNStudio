package com.neuraltrainer.core

/**
 * Activation functions available in the neural network
 */
enum class ActivationType(val nativeId: Int) {
    RELU(0),
    LEAKY_RELU(1),
    TANH(2),
    SIGMOID(3),
    GELU(4),
    SWISH(5),
    LINEAR(6);

    companion object {
        fun fromId(id: Int): ActivationType {
            return values().find { it.nativeId == id } ?: RELU
        }
    }
}

/**
 * Loss functions available for training
 */
enum class LossType(val nativeId: Int) {
    MSE(0),
    BCE(1),
    MAE(2),
    CROSS_ENTROPY(3),
    HUBER(4);

    companion object {
        fun fromId(id: Int): LossType {
            return values().find { it.nativeId == id } ?: MSE
        }
    }
}

/**
 * Optimizer algorithms for training
 */
enum class OptimizerType(val nativeId: Int) {
    SGD(0),
    MOMENTUM(1),
    RMSPROP(2),
    ADAM(3),
    ADAMW(4);

    companion object {
        fun fromId(id: Int): OptimizerType {
            return values().find { it.nativeId == id } ?: SGD
        }
    }
}

/**
 * Training tasks available in the app
 */
enum class TaskType(val name: String, val inputSize: Int, val outputSize: Int, val hiddenSize: Int = 16) {
    XOR("XOR", 2, 1, 8),
    AND("AND Gate", 2, 1, 8),
    OR("OR Gate", 2, 1, 8),
    NAND("NAND Gate", 2, 1, 8),
    SEVEN_SEGMENT("7-Segment Display", 4, 7, 32),
    PARITY("Parity Check", 4, 1, 16),
    SINE("Sine Wave", 1, 1, 24),
    CIRCLE("Circle Classification", 2, 1, 16),
    SPIRAL("Spiral Classification", 2, 2, 48),
    AUTOENCODER("Autoencoder", 8, 8, 16);

    companion object {
        fun fromPosition(position: Int): TaskType {
            return values()[position]
        }

        fun getNames(): Array<String> {
            return values().map { it.name }.toTypedArray()
        }
    }
}

/**
 * Data class representing a training sample
 */
data class TrainingSample(
    val inputs: FloatArray,
    val targets: FloatArray
)

/**
 * Data class representing a complete dataset
 */
data class Dataset(
    val samples: List<TrainingSample>,
    val inputSize: Int,
    val outputSize: Int
) {
    val size: Int get() = samples.size

    fun getInputMatrix(): Array<FloatArray> {
        return samples.map { it.inputs }.toTypedArray()
    }

    fun getTargetMatrix(): Array<FloatArray> {
        return samples.map { it.targets }.toTypedArray()
    }
}

/**
 * Object containing dataset generators for different tasks
 */
object DatasetGenerator {

    /**
     * Generate dataset for the specified task
     */
    fun generateDataset(task: TaskType): Dataset {
        return when (task) {
            TaskType.XOR -> generateXOR()
            TaskType.AND -> generateAND()
            TaskType.OR -> generateOR()
            TaskType.NAND -> generateNAND()
            TaskType.SEVEN_SEGMENT -> generateSevenSegment()
            TaskType.PARITY -> generateParity()
            TaskType.SINE -> generateSine()
            TaskType.CIRCLE -> generateCircle()
            TaskType.SPIRAL -> generateSpiral()
            TaskType.AUTOENCODER -> generateAutoencoder()
        }
    }

    private fun generateXOR(): Dataset {
        val samples = listOf(
            TrainingSample(floatArrayOf(0f, 0f), floatArrayOf(0f)),
            TrainingSample(floatArrayOf(0f, 1f), floatArrayOf(1f)),
            TrainingSample(floatArrayOf(1f, 0f), floatArrayOf(1f)),
            TrainingSample(floatArrayOf(1f, 1f), floatArrayOf(0f))
        )
        return Dataset(samples, 2, 1)
    }

    private fun generateAND(): Dataset {
        val samples = listOf(
            TrainingSample(floatArrayOf(0f, 0f), floatArrayOf(0f)),
            TrainingSample(floatArrayOf(0f, 1f), floatArrayOf(0f)),
            TrainingSample(floatArrayOf(1f, 0f), floatArrayOf(0f)),
            TrainingSample(floatArrayOf(1f, 1f), floatArrayOf(1f))
        )
        return Dataset(samples, 2, 1)
    }

    private fun generateOR(): Dataset {
        val samples = listOf(
            TrainingSample(floatArrayOf(0f, 0f), floatArrayOf(0f)),
            TrainingSample(floatArrayOf(0f, 1f), floatArrayOf(1f)),
            TrainingSample(floatArrayOf(1f, 0f), floatArrayOf(1f)),
            TrainingSample(floatArrayOf(1f, 1f), floatArrayOf(1f))
        )
        return Dataset(samples, 2, 1)
    }

    private fun generateNAND(): Dataset {
        val samples = listOf(
            TrainingSample(floatArrayOf(0f, 0f), floatArrayOf(1f)),
            TrainingSample(floatArrayOf(0f, 1f), floatArrayOf(1f)),
            TrainingSample(floatArrayOf(1f, 0f), floatArrayOf(1f)),
            TrainingSample(floatArrayOf(1f, 1f), floatArrayOf(0f))
        )
        return Dataset(samples, 2, 1)
    }

    private fun generateSevenSegment(): Dataset {
        // 4-bit input representing digits 0-9, 7-bit output for segments a-g
        val segmentMap = mapOf(
            0 to floatArrayOf(1f, 1f, 1f, 1f, 1f, 1f, 0f), // a,b,c,d,e,f
            1 to floatArrayOf(0f, 1f, 1f, 0f, 0f, 0f, 0f), // b,c
            2 to floatArrayOf(1f, 1f, 0f, 1f, 1f, 0f, 1f), // a,b,d,e,g
            3 to floatArrayOf(1f, 1f, 1f, 1f, 0f, 0f, 1f), // a,b,c,d,g
            4 to floatArrayOf(0f, 1f, 1f, 0f, 0f, 1f, 1f), // b,c,f,g
            5 to floatArrayOf(1f, 0f, 1f, 1f, 0f, 1f, 1f), // a,c,d,f,g
            6 to floatArrayOf(1f, 0f, 1f, 1f, 1f, 1f, 1f), // a,c,d,e,f,g
            7 to floatArrayOf(1f, 1f, 1f, 0f, 0f, 0f, 0f), // a,b,c
            8 to floatArrayOf(1f, 1f, 1f, 1f, 1f, 1f, 1f), // all
            9 to floatArrayOf(1f, 1f, 1f, 1f, 0f, 1f, 1f)  // a,b,c,d,f,g
        )

        val samples = mutableListOf<TrainingSample>()
        for (digit in 0..9) {
            val binaryInput = floatArrayOf(
                ((digit shr 3) and 1).toFloat(),
                ((digit shr 2) and 1).toFloat(),
                ((digit shr 1) and 1).toFloat(),
                (digit and 1).toFloat()
            )
            samples.add(TrainingSample(binaryInput, segmentMap[digit]!!))
        }
        return Dataset(samples, 4, 7)
    }

    private fun generateParity(): Dataset {
        val samples = mutableListOf<TrainingSample>()
        for (i in 0..15) {
            val bits = floatArrayOf(
                ((i shr 3) and 1).toFloat(),
                ((i shr 2) and 1).toFloat(),
                ((i shr 1) and 1).toFloat(),
                (i and 1).toFloat()
            )
            val parity = bits.sum() % 2
            samples.add(TrainingSample(bits, floatArrayOf(parity.toFloat())))
        }
        return Dataset(samples, 4, 1)
    }

    private fun generateSine(): Dataset {
        val samples = mutableListOf<TrainingSample>()
        for (i in 0..39) {
            val x = (i / 40.0f) * 2f * Math.PI.toFloat()
            val y = Math.sin(x.toDouble()).toFloat()
            samples.add(TrainingSample(floatArrayOf(x), floatArrayOf(y)))
        }
        return Dataset(samples, 1, 1)
    }

    private fun generateCircle(): Dataset {
        val samples = mutableListOf<TrainingSample>()
        val radius = 0.5f
        
        // Generate points inside circle
        for (i in 0..49) {
            val angle = (i / 50.0f) * 2f * Math.PI.toFloat()
            val r = (Math.random().toFloat() * radius)
            val x = r * Math.cos(angle.toDouble()).toFloat()
            val y = r * Math.sin(angle.toDouble()).toFloat()
            samples.add(TrainingSample(floatArrayOf(x, y), floatArrayOf(1f)))
        }
        
        // Generate points outside circle
        for (i in 0..49) {
            val angle = (i / 50.0f) * 2f * Math.PI.toFloat()
            val r = radius + Math.random().toFloat() * 0.3f
            val x = r * Math.cos(angle.toDouble()).toFloat()
            val y = r * Math.sin(angle.toDouble()).toFloat()
            samples.add(TrainingSample(floatArrayOf(x, y), floatArrayOf(0f)))
        }
        
        return Dataset(samples, 2, 1)
    }

    private fun generateSpiral(): Dataset {
        val samples = mutableListOf<TrainingSample>()
        val nPoints = 100
        
        // Generate two spiral classes
        for (i in 0 until nPoints) {
            val t = (i / nPoints.toFloat()) * 4f * Math.PI.toFloat()
            val r = (i / nPoints.toFloat())
            
            // Class 0
            val angle0 = t
            val x0 = (r * Math.cos(angle0.toDouble()) + Math.random().toFloat() * 0.1f - 0.05f).toFloat()
            val y0 = (r * Math.sin(angle0.toDouble()) + Math.random().toFloat() * 0.1f - 0.05f).toFloat()
            samples.add(TrainingSample(floatArrayOf(x0, y0), floatArrayOf(1f, 0f)))
            
            // Class 1
            val angle1 = t + Math.PI.toFloat()
            val x1 = (r * Math.cos(angle1.toDouble()) + Math.random().toFloat() * 0.1f - 0.05f).toFloat()
            val y1 = (r * Math.sin(angle1.toDouble()) + Math.random().toFloat() * 0.1f - 0.05f).toFloat()
            samples.add(TrainingSample(floatArrayOf(x1, y1), floatArrayOf(0f, 1f)))
        }
        
        return Dataset(samples, 2, 2)
    }

    private fun generateAutoencoder(): Dataset {
        val samples = mutableListOf<TrainingSample>()
        
        // Generate one-hot encoded patterns
        for (i in 0..7) {
            val input = FloatArray(8) { if (it == i) 1f else 0f }
            samples.add(TrainingSample(input.copyOf(), input.copyOf()))
        }
        
        return Dataset(samples, 8, 8)
    }
}
