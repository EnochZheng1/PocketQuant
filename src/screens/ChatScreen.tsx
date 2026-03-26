import React, {useState, useRef, useCallback, useEffect} from 'react';
import {
  View,
  FlatList,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  Text,
  TouchableOpacity,
} from 'react-native';
import type {Message} from '../types';
import {useLlama} from '../hooks/useLlama';
import MessageBubble from '../components/MessageBubble';
import InputBar from '../components/InputBar';
import BenchmarkOverlay from '../components/BenchmarkOverlay';

// Default model path on device (app-specific external files dir)
const DEFAULT_MODEL_PATH =
  '/storage/emulated/0/Android/data/com.remotellm/files/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf';

let nextId = 1;
function genId() {
  return String(nextId++);
}

export default function ChatScreen() {
  const {status, streamingText, benchmarks, loadModel, generate, stop} =
    useLlama();
  const [messages, setMessages] = useState<Message[]>([]);
  const [showBenchmark, setShowBenchmark] = useState(false);
  const flatListRef = useRef<FlatList>(null);
  const assistantIdRef = useRef<string | null>(null);

  // Auto-load model on mount
  useEffect(() => {
    loadModel(DEFAULT_MODEL_PATH);
  }, [loadModel]);

  // Update the streaming assistant message as tokens arrive
  useEffect(() => {
    if (assistantIdRef.current && streamingText) {
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantIdRef.current
            ? {...m, content: streamingText}
            : m,
        ),
      );
    }
  }, [streamingText]);

  // When generation completes, finalize the message
  useEffect(() => {
    if (status === 'ready' && assistantIdRef.current) {
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

  const handleSend = useCallback(
    (text: string) => {
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
      setMessages(prev => [...prev, userMsg, assistantMsg]);
      scrollToEnd();

      // Send to native engine
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

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
      keyboardVerticalOffset={0}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>PocketQuant</Text>
        <Text style={styles.headerStatus}>
          {status === 'ready'
            ? 'Ready'
            : status === 'generating'
            ? 'Generating...'
            : status === 'loading'
            ? 'Loading model...'
            : status === 'error'
            ? 'Error loading model'
            : 'No model loaded'}
        </Text>
      </View>

      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderItem}
        keyExtractor={item => item.id}
        style={styles.messageList}
        contentContainerStyle={styles.messageListContent}
        onContentSizeChange={() => scrollToEnd()}
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
    </KeyboardAvoidingView>
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
  headerStatus: {
    color: '#8E8E93',
    fontSize: 12,
    marginTop: 2,
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
