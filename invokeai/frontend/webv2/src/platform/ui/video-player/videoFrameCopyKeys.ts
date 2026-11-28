export type VideoFrameCopyFailureReason =
  | 'clipboard-failed'
  | 'draw-failed'
  | 'encode-failed'
  | 'not-ready'
  | 'stale'
  | 'unsupported';

export type VideoFrameCopyResult = { ok: true } | { ok: false; reason: VideoFrameCopyFailureReason };

export const VIDEO_FRAME_COPY_FAILURE_KEYS: Record<VideoFrameCopyFailureReason, string> = {
  'clipboard-failed': 'widgets.preview.copyCurrentFrameWriteFailed',
  'draw-failed': 'widgets.preview.copyCurrentFrameDrawFailed',
  'encode-failed': 'widgets.preview.copyCurrentFrameEncodeFailed',
  'not-ready': 'widgets.preview.copyCurrentFrameNotReady',
  stale: 'widgets.preview.copyCurrentFrameStale',
  unsupported: 'widgets.preview.copyCurrentFrameUnsupported',
};
