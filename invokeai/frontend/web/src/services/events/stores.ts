import type { EphemeralProgressImage } from 'features/controlLayers/store/types';
import type { ProgressImage } from 'features/nodes/types/common';
import { round } from 'lodash-es';
import type { WritableAtom } from 'nanostores';
import { atom, computed, map } from 'nanostores';
import { useEffect, useState } from 'react';
import type { ImageDTO, S } from 'services/api/types';
import type { AppSocket } from 'services/events/types';
import type { ManagerOptions, SocketOptions } from 'socket.io-client';

export const $socket = atom<AppSocket | null>(null);
export const $socketOptions = map<Partial<ManagerOptions & SocketOptions>>({});
export const $isConnected = atom<boolean>(false);
export const $lastProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);

export type ProgressAndResult = {
  sessionId: string;
  isFinished: boolean;
  progressImage?: ProgressImage;
  resultImage?: ImageDTO;
};
export const $progressImages = map({} as Record<string, ProgressAndResult>);

export type ProgressData = {
  sessionId: string;
  progressEvent: S['InvocationProgressEvent'] | null;
  progressImage: ProgressImage | null;
};

export const useProgressData = (
  $progressData: WritableAtom<Record<string, ProgressData>>,
  sessionId: string
): ProgressData => {
  const [value, setValue] = useState<ProgressData>(() => {
    return $progressData.get()[sessionId] ?? { sessionId, progressEvent: null, progressImage: null };
  });
  useEffect(() => {
    const unsub = $progressData.subscribe((data) => {
      const progressData = data[sessionId];
      if (!progressData) {
        return;
      }
      setValue(progressData);
    });
    return () => {
      unsub();
    };
  }, [$progressData, sessionId]);

  return value;
};

export const useHasProgressImage = (
  $progressData: WritableAtom<Record<string, ProgressData>>,
  sessionId: string
): boolean => {
  const [value, setValue] = useState(false);
  useEffect(() => {
    const unsub = $progressData.subscribe((data) => {
      const progressData = data[sessionId];
      setValue(Boolean(progressData?.progressImage));
    });
    return () => {
      unsub();
    };
  }, [$progressData, sessionId]);

  return value;
};

export const setProgress = (
  $progressData: WritableAtom<Record<string, ProgressData>>,
  data: S['InvocationProgressEvent']
) => {
  const progressData = $progressData.get();
  const current = progressData[data.session_id];
  if (current) {
    const next = { ...current };
    next.progressEvent = data;
    if (data.image) {
      next.progressImage = data.image;
    }
    $progressData.set({
      ...progressData,
      [data.session_id]: next,
    });
  } else {
    $progressData.set({
      ...progressData,
      [data.session_id]: {
        sessionId: data.session_id,
        progressEvent: data,
        progressImage: data.image ?? null,
      },
    });
  }
};

export const clearProgressEvent = ($progressData: WritableAtom<Record<string, ProgressData>>, sessionId: string) => {
  const progressData = $progressData.get();
  const current = progressData[sessionId];
  if (!current) {
    return;
  }
  const next = { ...current };
  next.progressEvent = null;
  $progressData.set({
    ...progressData,
    [sessionId]: next,
  });
};

export const clearProgressImage = ($progressData: WritableAtom<Record<string, ProgressData>>, sessionId: string) => {
  const progressData = $progressData.get();
  const current = progressData[sessionId];
  if (!current) {
    return;
  }
  const next = { ...current };
  next.progressImage = null;
  $progressData.set({
    ...progressData,
    [sessionId]: next,
  });
};

export const $lastCanvasProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $lastCanvasProgressImage = atom<EphemeralProgressImage | null>(null);
export const $lastWorkflowsProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $lastWorkflowsProgressImage = atom<EphemeralProgressImage | null>(null);
export const $lastUpscalingProgressEvent = atom<S['InvocationProgressEvent'] | null>(null);
export const $lastUpscalingProgressImage = atom<EphemeralProgressImage | null>(null);

export const $lastProgressImage = computed($lastProgressEvent, (val) => val?.image ?? null);
export const $hasLastProgressImage = computed($lastProgressEvent, (val) => Boolean(val?.image));
export const $lastProgressMessage = computed($lastProgressEvent, (val) => {
  if (!val) {
    return null;
  }

  let message = val.message;
  if (val.percentage) {
    message += ` (${round(val.percentage * 100)}%)`;
  }
  return message;
});
