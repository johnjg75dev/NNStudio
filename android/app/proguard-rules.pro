# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /usr/local/Cellar/android-sdk/24.3.3/tools/proguard/proguard-android.txt

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep Kotlin coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}

# Keep Gson serialization
-keepattributes Signature
-keepattributes *Annotation*
-dontobfuscate
-allowaccessmodification

# Keep our core classes
-keep class com.neuraltrainer.core.** { *; }
-keep class com.neuraltrainer.ui.** { *; }
