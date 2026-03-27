import {useEffect, useRef, useState, useCallback} from 'react';
import {NativeEventEmitter, NativeModules} from 'react-native';
import NativeLlama from '../../specs/NativeLlama';
import type {ModelStatus, BenchmarkStats} from '../types';

export function useLlama() {
  const [status, setStatus] = useState<ModelStatus>('unloaded');
  const [streamingText, setStreamingText] = useState('');
  const [benchmarks, setBenchmarks] = useState<BenchmarkStats | null>(null);
  const textRef = useRef('');
  const throttleRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const emitter = new NativeEventEmitter(NativeModules.LlamaModule);
    const sub = emitter.addListener('onToken', event => {
      if (event.error) {
        setStatus('error');
        return;
      }

      if (event.done) {
        // Generation complete — flush any pending throttled update
        if (throttleRef.current) {
          clearTimeout(throttleRef.current);
          throttleRef.current = null;
        }
        setStreamingText(textRef.current);
        setBenchmarks({
          tokensPerSecond: event.tokensPerSecond ?? 0,
          memoryMB: event.memoryMB ?? 0,
          totalTokens: event.totalTokens ?? 0,
          timeToFirstToken: event.timeToFirstTokenMs ?? 0,
          compressionRatio: event.compressionRatio ?? 0,
          fallbackRate: event.fallbackRate ?? 0,
        });
        setStatus('ready');
      } else {
        // Append token, throttle UI updates to ~20fps
        textRef.current += event.token;
        if (!throttleRef.current) {
          throttleRef.current = setTimeout(() => {
            setStreamingText(textRef.current);
            throttleRef.current = null;
          }, 50);
        }
      }
    });

    return () => {
      sub.remove();
      if (throttleRef.current) {
        clearTimeout(throttleRef.current);
      }
    };
  }, []);

  const loadModel = useCallback(async (path: string) => {
    setStatus('loading');
    console.log('[PocketQuant] Loading model from:', path);
    try {
      await NativeLlama.loadModel(path);
      console.log('[PocketQuant] Model loaded successfully');
      setStatus('ready');
      return true;
    } catch (e: any) {
      console.error('[PocketQuant] Failed to load model:', e?.message || e);
      setStatus('error');
      return false;
    }
  }, []);

  const generate = useCallback((prompt: string, maxTokens = 512) => {
    textRef.current = '';
    setStreamingText('');
    setBenchmarks(null);
    setStatus('generating');
    NativeLlama.startGeneration(prompt, maxTokens);
  }, []);

  const stop = useCallback(() => {
    NativeLlama.stopGeneration();
  }, []);

  const unload = useCallback(() => {
    NativeLlama.unloadModel();
    setStatus('unloaded');
  }, []);

  const setTurboMode = useCallback((enabled: boolean) => {
    NativeLlama.setTurboMode(enabled);
  }, []);

  return {status, streamingText, benchmarks, loadModel, generate, stop, unload, setTurboMode};
}
