export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  isStreaming?: boolean;
}

export interface BenchmarkStats {
  tokensPerSecond: number;
  memoryMB: number;
  totalTokens: number;
  timeToFirstToken: number;
  compressionRatio: number;
  fallbackRate: number;
}

export type ModelStatus = 'unloaded' | 'loading' | 'ready' | 'generating' | 'error';
