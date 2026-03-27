import type {TurboModule} from 'react-native';
import {TurboModuleRegistry} from 'react-native';

export interface Spec extends TurboModule {
  // Backend initialization
  initBackend(nativeLibDir: string): void;

  // Model management
  loadModel(modelPath: string): Promise<boolean>;
  unloadModel(): void;
  getModelStatus(): string;

  // Generation
  startGeneration(prompt: string, maxTokens: number): void;
  stopGeneration(): void;

  // TurboQuant compression
  setTurboMode(enabled: boolean): void;

  // Event lifecycle (required for NativeEventEmitter)
  addListener(eventName: string): void;
  removeListeners(count: number): void;
}

export default TurboModuleRegistry.getEnforcing<Spec>('LlamaModule');
