import React, {useCallback} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ToastAndroid,
} from 'react-native';
import Clipboard from '../utils/clipboard';
import type {Message} from '../types';

interface Props {
  message: Message;
}

function MessageBubble({message}: Props) {
  const isUser = message.role === 'user';

  const handleLongPress = useCallback(() => {
    if (!message.content) return;
    Clipboard.setString(message.content);
    ToastAndroid.show('Copied to clipboard', ToastAndroid.SHORT);
  }, [message.content]);

  return (
    <TouchableOpacity
      onLongPress={handleLongPress}
      activeOpacity={0.8}
      delayLongPress={400}
      style={[
        styles.container,
        isUser ? styles.userContainer : styles.assistantContainer,
      ]}>
      <View
        style={[
          styles.bubble,
          isUser ? styles.userBubble : styles.assistantBubble,
        ]}>
        <Text
          style={[styles.text, isUser ? styles.userText : styles.assistantText]}
          selectable>
          {message.content}
          {message.isStreaming && <Text style={styles.cursor}>|</Text>}
        </Text>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  userContainer: {
    alignItems: 'flex-end',
  },
  assistantContainer: {
    alignItems: 'flex-start',
  },
  bubble: {
    maxWidth: '80%',
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 16,
  },
  userBubble: {
    backgroundColor: '#007AFF',
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: '#2C2C2E',
    borderBottomLeftRadius: 4,
  },
  text: {
    fontSize: 16,
    lineHeight: 22,
  },
  userText: {
    color: '#FFFFFF',
  },
  assistantText: {
    color: '#E5E5EA',
  },
  cursor: {
    color: '#007AFF',
    fontWeight: '300',
  },
});

export default React.memo(MessageBubble);
