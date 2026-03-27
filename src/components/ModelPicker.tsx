import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  Modal,
  TouchableOpacity,
  ActivityIndicator,
  Switch,
} from 'react-native';
import type {ModelStatus} from '../types';
import type {DownloadState, DownloadFile} from '../hooks/useModelDownloader';

export interface ModelOption {
  id: string;
  name: string;
  size: string;
  filename: string; // first file (used to load)
  files: DownloadFile[];
  description: string;
}

const HF = 'https://huggingface.co';

export const AVAILABLE_MODELS: ModelOption[] = [
  {
    id: 'tinyllama-1.1b',
    name: 'TinyLlama 1.1B',
    size: '638 MB',
    filename: 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    files: [
      {
        url: `${HF}/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`,
        filename: 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
      },
    ],
    description: 'Fast, lightweight. Great for testing.',
  },
  {
    id: 'qwen-7b',
    name: 'Qwen2.5 7B',
    size: '4.4 GB',
    filename: 'qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf',
    files: [
      {
        url: `${HF}/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf`,
        filename: 'qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf',
      },
      {
        url: `${HF}/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf`,
        filename: 'qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf',
      },
    ],
    description: 'Balanced quality and speed.',
  },
  {
    id: 'qwen-14b',
    name: 'Qwen2.5 14B',
    size: '8.5 GB',
    filename: 'qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf',
    files: [
      {
        url: `${HF}/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf`,
        filename: 'qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf',
      },
      {
        url: `${HF}/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf`,
        filename: 'qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf',
      },
      {
        url: `${HF}/Qwen/Qwen2.5-14B-Instruct-GGUF/resolve/main/qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf`,
        filename: 'qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf',
      },
    ],
    description: 'High quality. Needs 12+ GB RAM.',
  },
];

interface Props {
  visible: boolean;
  onClose: () => void;
  onSelect: (modelId: string) => void;
  onDownload: (modelId: string) => void;
  onCancelDownload: () => void;
  currentModelId: string | null;
  modelStatus: ModelStatus;
  downloadState: DownloadState;
  downloadedModels: Set<string>;
  turboEnabled: boolean;
  onToggleTurbo: (enabled: boolean) => void;
}

function ProgressBar({progress}: {progress: number}) {
  return (
    <View style={progressStyles.track}>
      <View style={[progressStyles.fill, {width: `${Math.round(progress * 100)}%`}]} />
    </View>
  );
}

const progressStyles = StyleSheet.create({
  track: {
    height: 4,
    backgroundColor: '#3A3A3C',
    borderRadius: 2,
    marginTop: 8,
    overflow: 'hidden',
  },
  fill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 2,
  },
});

export default function ModelPicker({
  visible,
  onClose,
  onSelect,
  onDownload,
  onCancelDownload,
  currentModelId,
  modelStatus,
  downloadState,
  downloadedModels,
  turboEnabled,
  onToggleTurbo,
}: Props) {
  const isLoading = modelStatus === 'loading';

  return (
    <Modal
      visible={visible}
      transparent
      animationType="slide"
      onRequestClose={onClose}>
      <View style={styles.overlay}>
        <View style={styles.sheet}>
          <View style={styles.handle} />
          <Text style={styles.title}>Select Model</Text>
          <Text style={styles.subtitle}>
            Tap to download and run. Models are saved for offline use.
          </Text>

          {AVAILABLE_MODELS.map(model => {
            const isActive = currentModelId === model.id;
            const isDownloaded = downloadedModels.has(model.id);
            const isDownloading = downloadState.modelId === model.id;
            const isLoadingThis = isActive && isLoading;

            return (
              <TouchableOpacity
                key={model.id}
                style={[styles.modelCard, isActive && styles.modelCardActive]}
                onPress={() => {
                  if (isDownloading) return;
                  if (isLoading) return;
                  if (isDownloaded) {
                    onSelect(model.id);
                  } else {
                    onDownload(model.id);
                  }
                }}
                disabled={isDownloading || isLoading}
                activeOpacity={0.7}>
                <View style={styles.modelInfo}>
                  <View style={styles.modelHeader}>
                    <Text style={styles.modelName}>{model.name}</Text>
                    <Text style={styles.modelSize}>{model.size}</Text>
                  </View>
                  <Text style={styles.modelDesc}>{model.description}</Text>

                  {isDownloading && (
                    <View>
                      <ProgressBar progress={downloadState.progress} />
                      <View style={styles.downloadRow}>
                        <Text style={styles.downloadText}>
                          {downloadState.downloadedMB} / {downloadState.totalMB || '?'} MB
                          {downloadState.totalFiles > 1 &&
                            ` (file ${downloadState.currentFile}/${downloadState.totalFiles})`}
                        </Text>
                        <TouchableOpacity onPress={onCancelDownload}>
                          <Text style={styles.cancelText}>Cancel</Text>
                        </TouchableOpacity>
                      </View>
                    </View>
                  )}
                </View>

                {!isDownloading && (
                  <View style={styles.statusIcon}>
                    {isLoadingThis ? (
                      <ActivityIndicator size="small" color="#007AFF" />
                    ) : isActive ? (
                      <Text style={styles.checkmark}>✓</Text>
                    ) : isDownloaded ? (
                      <Text style={styles.readyDot}>●</Text>
                    ) : (
                      <Text style={styles.downloadIcon}>↓</Text>
                    )}
                  </View>
                )}
              </TouchableOpacity>
            );
          })}

          <View style={styles.turboRow}>
            <View>
              <Text style={styles.turboLabel}>TurboQuant 1-bit Cache</Text>
              <Text style={styles.turboDesc}>
                {turboEnabled ? '13x compression, 4x context' : 'Standard KV cache'}
              </Text>
            </View>
            <Switch
              value={turboEnabled}
              onValueChange={onToggleTurbo}
              trackColor={{false: '#3A3A3C', true: '#30D158'}}
              thumbColor="#FFFFFF"
            />
          </View>

          <TouchableOpacity style={styles.closeButton} onPress={onClose}>
            <Text style={styles.closeText}>Close</Text>
          </TouchableOpacity>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    justifyContent: 'flex-end',
  },
  sheet: {
    backgroundColor: '#1C1C1E',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingHorizontal: 20,
    paddingBottom: 34,
  },
  handle: {
    width: 36,
    height: 5,
    borderRadius: 3,
    backgroundColor: '#636366',
    alignSelf: 'center',
    marginTop: 10,
    marginBottom: 16,
  },
  title: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 4,
  },
  subtitle: {
    color: '#8E8E93',
    fontSize: 13,
    marginBottom: 16,
  },
  modelCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2C2C2E',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderWidth: 1.5,
    borderColor: 'transparent',
  },
  modelCardActive: {
    borderColor: '#007AFF',
  },
  modelInfo: {
    flex: 1,
  },
  modelHeader: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 4,
  },
  modelName: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    marginRight: 8,
  },
  modelSize: {
    color: '#8E8E93',
    fontSize: 13,
  },
  modelDesc: {
    color: '#AEAEB2',
    fontSize: 13,
  },
  downloadRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 4,
  },
  downloadText: {
    color: '#8E8E93',
    fontSize: 11,
  },
  cancelText: {
    color: '#FF453A',
    fontSize: 12,
    fontWeight: '600',
  },
  statusIcon: {
    marginLeft: 10,
    width: 28,
    alignItems: 'center',
  },
  checkmark: {
    color: '#007AFF',
    fontSize: 20,
    fontWeight: '700',
  },
  readyDot: {
    color: '#30D158',
    fontSize: 12,
  },
  downloadIcon: {
    color: '#8E8E93',
    fontSize: 20,
    fontWeight: '700',
  },
  turboRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#2C2C2E',
    borderRadius: 12,
    padding: 14,
    marginTop: 6,
    marginBottom: 4,
  },
  turboLabel: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '600',
  },
  turboDesc: {
    color: '#8E8E93',
    fontSize: 12,
    marginTop: 2,
  },
  closeButton: {
    alignSelf: 'center',
    marginTop: 8,
    paddingVertical: 10,
    paddingHorizontal: 24,
  },
  closeText: {
    color: '#8E8E93',
    fontSize: 16,
  },
});
