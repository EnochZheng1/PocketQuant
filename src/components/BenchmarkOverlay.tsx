import React from 'react';
import {View, Text, StyleSheet, TouchableOpacity} from 'react-native';
import type {BenchmarkStats} from '../types';

interface Props {
  stats: BenchmarkStats | null;
  visible: boolean;
  onDismiss: () => void;
}

export default function BenchmarkOverlay({stats, visible, onDismiss}: Props) {
  if (!visible || !stats) {
    return null;
  }

  return (
    <TouchableOpacity
      style={styles.container}
      onPress={onDismiss}
      activeOpacity={0.9}>
      <View style={styles.card}>
        <Text style={styles.title}>Benchmark</Text>
        <View style={styles.row}>
          <Text style={styles.label}>Tokens/sec</Text>
          <Text style={styles.value}>{stats.tokensPerSecond.toFixed(1)}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Memory</Text>
          <Text style={styles.value}>{stats.memoryMB.toFixed(0)} MB</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Total tokens</Text>
          <Text style={styles.value}>{stats.totalTokens}</Text>
        </View>
        <View style={styles.row}>
          <Text style={styles.label}>Time to first token</Text>
          <Text style={styles.value}>{stats.timeToFirstToken.toFixed(0)} ms</Text>
        </View>
        <Text style={styles.hint}>Tap to dismiss</Text>
      </View>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 60,
    right: 12,
    zIndex: 100,
  },
  card: {
    backgroundColor: 'rgba(30, 30, 30, 0.95)',
    borderRadius: 12,
    padding: 14,
    minWidth: 180,
    borderWidth: 1,
    borderColor: '#3A3A3C',
  },
  title: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
    marginBottom: 8,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  label: {
    color: '#8E8E93',
    fontSize: 12,
  },
  value: {
    color: '#E5E5EA',
    fontSize: 12,
    fontWeight: '600',
    marginLeft: 12,
  },
  hint: {
    color: '#636366',
    fontSize: 10,
    textAlign: 'center',
    marginTop: 6,
  },
});
