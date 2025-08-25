import { useStore } from '@nanostores/react';
import { useVideoViewerContext } from 'features/video/context/VideoViewerContext';
import type { MediaStateOwner } from 'media-chrome/dist/media-store/state-mediator.js';
import { useCallback } from 'react';

export const useCaptureVideoFrame = () => {
  const { $videoRef } = useVideoViewerContext();
  const videoRef = useStore($videoRef);

  const captureFrame = useCallback(
    (video?: MediaStateOwner): File => {
      // Use provided video or fall back to context video ref
      const targetVideo = video || videoRef;
      // Validate video element
      if (!targetVideo || targetVideo.videoWidth === 0 || targetVideo.videoHeight === 0) {
        throw new Error('Invalid video element or video not loaded');
      }

      // Check if video is ready for capture
      if (targetVideo.readyState && targetVideo.readyState < 2) {
        // HAVE_CURRENT_DATA = 2
        throw new Error('Video is not ready for frame capture');
      }

      const canvas = document.createElement('canvas');
      canvas.width = targetVideo.videoWidth || 0;
      canvas.height = targetVideo.videoHeight || 0;

      const context = canvas.getContext('2d');
      if (!context) {
        throw new Error('Failed to get canvas 2D context');
      }

      // Draw the current video frame to canvas
      context.drawImage(targetVideo as HTMLVideoElement, 0, 0);

      // Convert to data URL with proper format
      const dataUri = canvas.toDataURL('image/png', 0.92);
      const data = dataUri.split(',')[1];
      const mimeType = dataUri.split(';')[0]?.slice(5);

      if (!data || !mimeType) {
        throw new Error('Failed to extract image data from canvas');
      }

      // Convert to blob
      const bytes = window.atob(data);
      const buf = new ArrayBuffer(bytes.length);
      const arr = new Uint8Array(buf);

      for (let i = 0; i < bytes.length; i++) {
        arr[i] = bytes.charCodeAt(i);
      }

      const blob = new Blob([arr], { type: mimeType });
      const file = new File([blob], 'frame.png', { type: mimeType });
      return file;
    },
    [videoRef]
  );

  return {
    captureFrame,
  };
};
