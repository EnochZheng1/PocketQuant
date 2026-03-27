import React, {useState, useRef, useCallback, useEffect} from 'react';
import {View, FlatList, StyleSheet, Text, TouchableOpacity} from 'react-native';
import {activateKeepAwake, deactivateKeepAwake} from 'react-native-keep-awake';
import type {Message} from '../types';
import {useLlama} from '../hooks/useLlama';
import {useModelDownloader} from '../hooks/useModelDownloader';
import MessageBubble from '../components/MessageBubble';
import InputBar from '../components/InputBar';
import BenchmarkOverlay from '../components/BenchmarkOverlay';
import ModelPicker, {AVAILABLE_MODELS} from '../components/ModelPicker';

let nextId = 1;
function genId() {
  return String(nextId++);
}

export default function ChatScreen() {
  const {status, streamingText, benchmarks, loadModel, generate, stop} =
    useLlama();
  const {downloadState, downloadModel, cancelDownload, isModelDownloaded, getModelPath} =
    useModelDownloader();
  const [messages, setMessages] = useState<Message[]>([]);
  const [showBenchmark, setShowBenchmark] = useState(false);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [currentModelId, setCurrentModelId] = useState<string | null>(null);
  const [downloadedModels, setDownloadedModels] = useState<Set<string>>(new Set());
  const flatListRef = useRef<FlatList>(null);
  const assistantIdRef = useRef<string | null>(null);
  const isGeneratingRef = useRef(false);

  // Check which models are already downloaded on mount
  useEffect(() => {
    (async () => {
      const downloaded = new Set<string>();
      for (const model of AVAILABLE_MODELS) {
        if (await isModelDownloaded(model.files)) {
          downloaded.add(model.id);
        }
      }
      setDownloadedModels(downloaded);
    })();
  }, [isModelDownloaded]);

  // Update the streaming assistant message as tokens arrive
  useEffect(() => {
    if (assistantIdRef.current && streamingText && isGeneratingRef.current) {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantIdRef.current
            ? {...m, content: streamingText}
            : m,
        ),
      );
      scrollToEnd();
    }
  }, [streamingText]);

  // Prevent screen timeout during inference
  useEffect(() => {
    if (status === 'generating') {
      activateKeepAwake();
    } else {
      deactivateKeepAwake();
    }
  }, [status]);

  // When generation completes, finalize the message
  useEffect(() => {
    if (
      status === 'ready' &&
      isGeneratingRef.current &&
      assistantIdRef.current
    ) {
      isGeneratingRef.current = false;
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantIdRef.current
            ? {...m, isStreaming: false}
            : m,
        ),
      );
      assistantIdRef.current = null;
      if (benchmarks) {
        setShowBenchmark(true);
      }
    }
  }, [status, benchmarks]);

  const scrollToEnd = useCallback(() => {
    setTimeout(() => {
      flatListRef.current?.scrollToEnd({animated: true});
    }, 50);
  }, []);

  // Download then auto-load
  const handleDownload = useCallback(
    async (modelId: string) => {
      const model = AVAILABLE_MODELS.find(m => m.id === modelId);
      if (!model) return;

      const success = await downloadModel(modelId, model.files);
      if (success) {
        setDownloadedModels(prev => new Set(prev).add(modelId));
        // Auto-load after download
        setCurrentModelId(modelId);
        setMessages([]);
        const modelPath = getModelPath(model.filename);
        const loaded = await loadModel(modelPath);
        if (loaded) {
          setShowModelPicker(false);
        }
      }
    },
    [downloadModel, getModelPath, loadModel],
  );

  // Load an already-downloaded model
  const handleSelectModel = useCallback(
    async (modelId: string) => {
      if (modelId === currentModelId && status === 'ready') {
        setShowModelPicker(false);
        return;
      }
      const model = AVAILABLE_MODELS.find(m => m.id === modelId);
      if (!model) return;

      setCurrentModelId(modelId);
      setMessages([]);
      const modelPath = getModelPath(model.filename);
      const success = await loadModel(modelPath);
      if (success) {
        setShowModelPicker(false);
      }
    },
    [currentModelId, status, loadModel, getModelPath],
  );

  const handleSend = useCallback(
    (text: string) => {
      if (isGeneratingRef.current) return;

      const userMsg: Message = {
        id: genId(),
        role: 'user',
        content: text,
        timestamp: Date.now(),
      };

      const assistantMsg: Message = {
        id: genId(),
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        isStreaming: true,
      };

      assistantIdRef.current = assistantMsg.id;
      isGeneratingRef.current = true;
      setMessages(prev => [...prev, userMsg, assistantMsg]);
      scrollToEnd();

      generate(text, 512);
    },
    [generate, scrollToEnd],
  );

  const handleStop = useCallback(() => {
    stop();
  }, [stop]);

  const renderItem = useCallback(
    ({item}: {item: Message}) => <MessageBubble message={item} />,
    [],
  );

  const currentModel = AVAILABLE_MODELS.find(m => m.id === currentModelId);
  const isDownloading = downloadState.modelId !== null;

  return (
    <View style={styles.container}>
      <TouchableOpacity
        style={styles.header}
        onPress={() => setShowModelPicker(true)}
        activeOpacity={0.7}>
        <Text style={styles.headerTitle}>PocketQuant</Text>
        <View style={styles.headerRow}>
          <Text style={styles.headerStatus}>
            {isDownloading
              ? `Downloading... ${downloadState.downloadedMB}/${downloadState.totalMB || '?'} MB`
              : currentModel
              ? `${currentModel.name} — `
              : ''}
            {!isDownloading &&
              (status === 'ready'
                ? 'Ready'
                : status === 'generating'
                ? 'Generating...'
                : status === 'loading'
                ? 'Loading model...'
                : status === 'error'
                ? 'Error'
                : 'Tap to select model')}
          </Text>
          <Text style={styles.headerChevron}>▾</Text>
        </View>
      </TouchableOpacity>

      {!currentModelId && !isDownloading && status !== 'loading' && (
        <TouchableOpacity
          style={styles.emptyState}
          onPress={() => setShowModelPicker(true)}>
          <Text style={styles.emptyIcon}>🧠</Text>
          <Text style={styles.emptyTitle}>No Model Loaded</Text>
          <Text style={styles.emptyDesc}>
            Tap here to download and run a model
          </Text>
        </TouchableOpacity>
      )}

      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderItem}
        keyExtractor={item => item.id}
        style={styles.messageList}
        contentContainerStyle={styles.messageListContent}
        onContentSizeChange={() => scrollToEnd()}
        keyboardShouldPersistTaps="handled"
      />

      {status === 'generating' && (
        <TouchableOpacity style={styles.stopButton} onPress={handleStop}>
          <Text style={styles.stopText}>Stop</Text>
        </TouchableOpacity>
      )}

      <BenchmarkOverlay
        stats={benchmarks}
        visible={showBenchmark}
        onDismiss={() => setShowBenchmark(false)}
      />

      <InputBar onSend={handleSend} modelStatus={status} />

      <ModelPicker
        visible={showModelPicker}
        onClose={() => setShowModelPicker(false)}
        onSelect={handleSelectModel}
        onDownload={handleDownload}
        onCancelDownload={cancelDownload}
        currentModelId={currentModelId}
        modelStatus={status}
        downloadState={downloadState}
        downloadedModels={downloadedModels}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  header: {
    paddingTop: 12,
    paddingBottom: 12,
    paddingHorizontal: 16,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#3A3A3C',
    backgroundColor: '#1C1C1E',
    alignItems: 'center',
  },
  headerTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
  },
  headerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 2,
  },
  headerStatus: {
    color: '#8E8E93',
    fontSize: 12,
  },
  headerChevron: {
    color: '#8E8E93',
    fontSize: 10,
    marginLeft: 4,
  },
  emptyState: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
  },
  emptyIcon: {
    fontSize: 48,
    marginBottom: 12,
  },
  emptyTitle: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '600',
    marginBottom: 8,
  },
  emptyDesc: {
    color: '#8E8E93',
    fontSize: 14,
    textAlign: 'center',
  },
  messageList: {
    flex: 1,
  },
  messageListContent: {
    paddingVertical: 12,
  },
  stopButton: {
    alignSelf: 'center',
    backgroundColor: '#FF3B30',
    borderRadius: 16,
    paddingHorizontal: 20,
    paddingVertical: 8,
    marginBottom: 8,
  },
  stopText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
});
