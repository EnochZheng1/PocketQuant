import {useState, useCallback, useRef} from 'react';
import {NativeModules} from 'react-native';
import RNFS from 'react-native-fs';

const MODELS_DIR = `${RNFS.DocumentDirectoryPath}/models`;

export interface DownloadFile {
  url: string;
  filename: string;
}

export interface DownloadState {
  modelId: string | null;
  progress: number; // 0-1
  downloadedMB: number;
  totalMB: number;
  currentFile: number;
  totalFiles: number;
  error: string | null;
}

const INITIAL_STATE: DownloadState = {
  modelId: null,
  progress: 0,
  downloadedMB: 0,
  totalMB: 0,
  currentFile: 0,
  totalFiles: 0,
  error: null,
};

export function useModelDownloader() {
  const [downloadState, setDownloadState] = useState<DownloadState>(INITIAL_STATE);
  const abortRef = useRef(false);
  const jobIdRef = useRef<number | null>(null);

  const isModelDownloaded = useCallback(async (files: DownloadFile[]): Promise<boolean> => {
    try {
      await RNFS.mkdir(MODELS_DIR);
      for (const file of files) {
        const exists = await RNFS.exists(`${MODELS_DIR}/${file.filename}`);
        if (!exists) return false;
      }
      return true;
    } catch {
      return false;
    }
  }, []);

  const downloadModel = useCallback(
    async (modelId: string, files: DownloadFile[]): Promise<boolean> => {
      abortRef.current = false;

      try {
        await RNFS.mkdir(MODELS_DIR);
      } catch {}

      setDownloadState({
        modelId,
        progress: 0,
        downloadedMB: 0,
        totalMB: 0,
        currentFile: 0,
        totalFiles: files.length,
        error: null,
      });

      let cumulativeBytes = 0;

      for (let i = 0; i < files.length; i++) {
        if (abortRef.current) {
          setDownloadState(INITIAL_STATE);
          return false;
        }

        const file = files[i];
        const destPath = `${MODELS_DIR}/${file.filename}`;

        // Skip if already downloaded
        const exists = await RNFS.exists(destPath);
        if (exists) {
          const stat = await RNFS.stat(destPath);
          cumulativeBytes += Number(stat.size);
          setDownloadState(prev => ({
            ...prev,
            currentFile: i + 1,
            downloadedMB: Math.round(cumulativeBytes / (1024 * 1024)),
          }));
          continue;
        }

        // Download with progress
        const prevCumulative = cumulativeBytes;
        const result = RNFS.downloadFile({
          fromUrl: file.url,
          toFile: destPath,
          background: true,
          discretionary: false,
          cacheable: false,
          progressInterval: 500,
          begin: (res) => {
            const totalForAllFiles = prevCumulative + res.contentLength;
            setDownloadState(prev => ({
              ...prev,
              currentFile: i + 1,
              totalMB: Math.round(totalForAllFiles / (1024 * 1024)),
            }));
          },
          progress: (res) => {
            const currentTotal = prevCumulative + res.bytesWritten;
            const overallTotal = prevCumulative + res.contentLength;
            setDownloadState(prev => ({
              ...prev,
              progress: overallTotal > 0 ? currentTotal / overallTotal : 0,
              downloadedMB: Math.round(currentTotal / (1024 * 1024)),
              totalMB: Math.round(overallTotal / (1024 * 1024)),
            }));
          },
        });

        jobIdRef.current = result.jobId;

        const downloadResult = await result.promise;
        jobIdRef.current = null;

        if (downloadResult.statusCode !== 200) {
          // Clean up partial file
          try { await RNFS.unlink(destPath); } catch {}
          setDownloadState(prev => ({
            ...prev,
            error: `Download failed (HTTP ${downloadResult.statusCode})`,
          }));
          return false;
        }

        cumulativeBytes += downloadResult.bytesWritten;
      }

      setDownloadState(prev => ({...prev, progress: 1}));
      return true;
    },
    [],
  );

  const cancelDownload = useCallback(() => {
    abortRef.current = true;
    if (jobIdRef.current !== null) {
      RNFS.stopDownload(jobIdRef.current);
      jobIdRef.current = null;
    }
    setDownloadState(INITIAL_STATE);
  }, []);

  const getModelPath = useCallback((filename: string) => {
    return `${MODELS_DIR}/${filename}`;
  }, []);

  return {downloadState, downloadModel, cancelDownload, isModelDownloaded, getModelPath};
}
