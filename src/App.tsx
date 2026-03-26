import React from 'react';
import {SafeAreaView, StatusBar, StyleSheet} from 'react-native';
import ChatScreen from './screens/ChatScreen';

export default function App() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#1C1C1E" />
      <ChatScreen />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
});
