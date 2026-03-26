import React, {useState} from 'react';
import {
  View,
  TextInput,
  TouchableOpacity,
  Text,
  StyleSheet,
} from 'react-native';
import type {ModelStatus} from '../types';

interface Props {
  onSend: (text: string) => void;
  modelStatus: ModelStatus;
}

export default function InputBar({onSend, modelStatus}: Props) {
  const [text, setText] = useState('');
  const disabled = modelStatus === 'generating' || modelStatus === 'loading';

  const handleSend = () => {
    const trimmed = text.trim();
    if (trimmed && !disabled) {
      onSend(trimmed);
      setText('');
    }
  };

  return (
    <View style={styles.container}>
      <TextInput
        style={styles.input}
        value={text}
        onChangeText={setText}
        placeholder={
          modelStatus === 'unloaded'
            ? 'Model not loaded...'
            : modelStatus === 'loading'
            ? 'Loading model...'
            : 'Type a message...'
        }
        placeholderTextColor="#8E8E93"
        multiline
        maxLength={2048}
        editable={!disabled}
        onSubmitEditing={handleSend}
        blurOnSubmit={false}
      />
      <TouchableOpacity
        style={[styles.sendButton, disabled && styles.sendButtonDisabled]}
        onPress={handleSend}
        disabled={disabled || !text.trim()}>
        <Text style={styles.sendText}>Send</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#3A3A3C',
    backgroundColor: '#1C1C1E',
  },
  input: {
    flex: 1,
    minHeight: 40,
    maxHeight: 120,
    backgroundColor: '#2C2C2E',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    fontSize: 16,
    color: '#FFFFFF',
    marginRight: 8,
  },
  sendButton: {
    backgroundColor: '#007AFF',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 10,
    justifyContent: 'center',
  },
  sendButtonDisabled: {
    opacity: 0.4,
  },
  sendText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});
