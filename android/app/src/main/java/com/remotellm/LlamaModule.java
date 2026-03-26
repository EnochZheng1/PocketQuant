package com.remotellm;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.facebook.react.bridge.Arguments;
import com.facebook.react.bridge.Promise;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.WritableMap;
import com.facebook.react.modules.core.DeviceEventManagerModule;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class LlamaModule extends NativeLlamaSpec {
    public static final String NAME = "LlamaModule";

    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean isGenerating = new AtomicBoolean(false);
    private final AtomicBoolean stopRequested = new AtomicBoolean(false);
    private boolean modelLoaded = false;
    private boolean backendInitialized = false;

    // JNI native methods
    private native void nativeInit(String nativeLibDir);
    private native int  nativeLoadModel(String path, int nThreads);
    private native int  nativeProcessSystemPrompt(String prompt);
    private native int  nativeProcessUserPrompt(String prompt, int maxTokens);
    private native String nativeGenerateNextToken();
    private native void nativeUnload();
    private native void nativeShutdown();
    private native float[] nativeGetModelInfo();

    static {
        System.loadLibrary("remotellm");
    }

    public LlamaModule(ReactApplicationContext context) {
        super(context);
    }

    @Override
    @NonNull
    public String getName() {
        return NAME;
    }

    @Override
    public void initBackend(String nativeLibDir) {
        if (!backendInitialized) {
            nativeInit(nativeLibDir);
            backendInitialized = true;
        }
    }

    @Override
    public void loadModel(String modelPath, Promise promise) {
        executor.execute(() -> {
            try {
                int nThreads = Math.max(2, Math.min(4,
                    Runtime.getRuntime().availableProcessors() - 2));

                // Initialize backend if not done yet
                if (!backendInitialized) {
                    String nativeLibDir = getReactApplicationContext()
                        .getApplicationInfo().nativeLibraryDir;
                    nativeInit(nativeLibDir);
                    backendInitialized = true;
                }

                // Unload previous model if any
                if (modelLoaded) {
                    nativeUnload();
                    modelLoaded = false;
                }

                int result = nativeLoadModel(modelPath, nThreads);
                if (result == 0) {
                    // Process a default system prompt
                    nativeProcessSystemPrompt("You are a helpful assistant.");
                    modelLoaded = true;
                    promise.resolve(true);
                } else {
                    promise.reject("LOAD_ERROR", "Failed to load model (error code: " + result + ")");
                }
            } catch (Exception e) {
                promise.reject("LOAD_ERROR", "Exception loading model: " + e.getMessage());
            }
        });
    }

    @Override
    public void startGeneration(String prompt, double maxTokens) {
        if (!modelLoaded || isGenerating.get()) return;

        isGenerating.set(true);
        stopRequested.set(false);

        executor.execute(() -> {
            try {
                long startTime = System.nanoTime();
                long firstTokenTime = 0;
                int tokenCount = 0;

                // Process user prompt
                int processResult = nativeProcessUserPrompt(prompt, (int) maxTokens);
                if (processResult != 0) {
                    WritableMap errorEvent = Arguments.createMap();
                    errorEvent.putBoolean("error", true);
                    errorEvent.putString("message", "Failed to process prompt");
                    sendEvent("onToken", errorEvent);
                    isGenerating.set(false);
                    return;
                }

                // Token generation loop
                while (!stopRequested.get()) {
                    String token = nativeGenerateNextToken();

                    // null = generation complete (EOG or stop position)
                    if (token == null) break;

                    // Empty string = partial UTF-8, skip emitting
                    if (token.isEmpty()) continue;

                    tokenCount++;

                    if (firstTokenTime == 0) {
                        firstTokenTime = System.nanoTime();
                    }

                    WritableMap event = Arguments.createMap();
                    event.putString("token", token);
                    event.putInt("tokenIndex", tokenCount);
                    sendEvent("onToken", event);
                }

                // Send completion event
                long elapsed = System.nanoTime() - startTime;
                double seconds = elapsed / 1_000_000_000.0;
                double ttft = firstTokenTime > 0
                    ? (firstTokenTime - startTime) / 1_000_000.0  // ms
                    : 0;

                WritableMap doneEvent = Arguments.createMap();
                doneEvent.putBoolean("done", true);
                doneEvent.putDouble("tokensPerSecond",
                    tokenCount / Math.max(seconds, 0.001));
                doneEvent.putInt("totalTokens", tokenCount);
                doneEvent.putDouble("timeToFirstTokenMs", ttft);
                doneEvent.putDouble("memoryMB",
                    android.os.Debug.getNativeHeapAllocatedSize() / (1024.0 * 1024.0));
                sendEvent("onToken", doneEvent);

            } catch (Exception e) {
                WritableMap errorEvent = Arguments.createMap();
                errorEvent.putBoolean("error", true);
                errorEvent.putString("message", e.getMessage());
                sendEvent("onToken", errorEvent);
            } finally {
                isGenerating.set(false);
            }
        });
    }

    @Override
    public void stopGeneration() {
        stopRequested.set(true);
    }

    @Override
    public void unloadModel() {
        executor.execute(() -> {
            if (modelLoaded) {
                nativeUnload();
                modelLoaded = false;
            }
        });
    }

    @Override
    public String getModelStatus() {
        if (isGenerating.get()) return "generating";
        if (modelLoaded) return "ready";
        return "unloaded";
    }

    private void sendEvent(String eventName, @Nullable WritableMap params) {
        getReactApplicationContext()
            .getJSModule(DeviceEventManagerModule.RCTDeviceEventEmitter.class)
            .emit(eventName, params);
    }

    @Override
    public void addListener(String eventName) {
        // Required for NativeEventEmitter
    }

    @Override
    public void removeListeners(double count) {
        // Required for NativeEventEmitter
    }
}
