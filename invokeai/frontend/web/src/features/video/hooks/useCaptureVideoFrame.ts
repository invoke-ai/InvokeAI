import { useCallback, useRef } from 'react';

export const useCaptureVideoFrame = () => {
  const videoRef = useRef<HTMLVideoElement>(null);

  const captureFrame = useCallback((video: HTMLVideoElement): File => {
    // Validate video element
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      throw new Error('Invalid video element or video not loaded');
    }

    // Check if video is ready for capture
    if (video.readyState < 2) {
      // HAVE_CURRENT_DATA = 2
      throw new Error('Video is not ready for frame capture');
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Failed to get canvas 2D context');
    }

    // Draw the current video frame to canvas
    context.drawImage(video, 0, 0);

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
  }, []);

  return {
    captureFrame,
    videoRef,
  };
};
