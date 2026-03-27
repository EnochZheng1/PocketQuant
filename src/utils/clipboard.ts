// React Native still ships Clipboard internally, just deprecated the export
// This wrapper avoids the deprecation warning
import {NativeModules} from 'react-native';

const NativeClipboard = NativeModules.Clipboard;

export default {
  setString(content: string) {
    NativeClipboard?.setString?.(content);
  },
};
