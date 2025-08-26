import { logger } from 'app/logging/logger';
import { toast } from 'features/toast/toast';
import { useCallback } from 'react';
import { serializeError } from 'serialize-error';
import { uploadImage } from 'services/api/endpoints/images';

const log = logger('video');

const captureFrame = (video: HTMLVideoElement): File => {
  // Validate video element
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    throw new Error('Invalid video element or video not loaded');
  }

  // Check if video is ready for capture
  // https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/readyState
  // 2 == HAVE_CURRENT_DATA
  if (video.readyState < 2) {
    throw new Error('Video is not ready for frame capture');
  }

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 0;
  canvas.height = video.videoHeight || 0;

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
};

export const useCaptureVideoFrame = () => {
  /*
   * Capture the current frame of the video uploading it as an asset.
   *
   * Toasts on success or failure. For convenience, accepts null but immediately creates a toast.
   */
  const captureVideoFrame = useCallback(async (video: HTMLVideoElement | null) => {
    try {
      if (!video) {
        toast({
          status: 'error',
          title: 'Video not ready',
          description: 'Please wait for the video to load before capturing a frame.',
        });
        return;
      }
      const file = captureFrame(video);
      await uploadImage({ file, image_category: 'user', is_intermediate: false, silent: true });
      toast({
        status: 'success',
        title: 'Frame saved to assets tab',
      });
    } catch (error) {
      log.error({ error: serializeError(error as Error) }, 'Failed to capture frame');
      toast({
        status: 'error',
        title: 'Failed to capture frame',
        description: 'There was an error capturing the current video frame.',
      });
    }
  }, []);

  return captureVideoFrame;
};
