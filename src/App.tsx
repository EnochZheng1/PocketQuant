import React from 'react';
import {StatusBar, StyleSheet, View} from 'react-native';
import {SafeAreaProvider, SafeAreaView} from 'react-native-safe-area-context';
import ChatScreen from './screens/ChatScreen';

export default function App() {
  return (
    <SafeAreaProvider>
      <View style={styles.container}>
        <StatusBar barStyle="light-content" backgroundColor="#1C1C1E" />
        <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
          <ChatScreen />
        </SafeAreaView>
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
});
