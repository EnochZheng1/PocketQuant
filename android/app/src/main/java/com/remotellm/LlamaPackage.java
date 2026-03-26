package com.remotellm;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

import com.facebook.react.TurboReactPackage;
import com.facebook.react.bridge.NativeModule;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.module.model.ReactModuleInfo;
import com.facebook.react.module.model.ReactModuleInfoProvider;

import java.util.HashMap;
import java.util.Map;

public class LlamaPackage extends TurboReactPackage {

    @Nullable
    @Override
    public NativeModule getModule(@NonNull String name, @NonNull ReactApplicationContext reactContext) {
        if (name.equals(LlamaModule.NAME)) {
            return new LlamaModule(reactContext);
        }
        return null;
    }

    @Override
    public ReactModuleInfoProvider getReactModuleInfoProvider() {
        return () -> {
            Map<String, ReactModuleInfo> moduleInfos = new HashMap<>();
            moduleInfos.put(LlamaModule.NAME, new ReactModuleInfo(
                LlamaModule.NAME,
                LlamaModule.NAME,
                false, // canOverrideExistingModule
                false, // needsEagerInit
                false, // isCxxModule
                true   // isTurboModule
            ));
            return moduleInfos;
        };
    }
}
