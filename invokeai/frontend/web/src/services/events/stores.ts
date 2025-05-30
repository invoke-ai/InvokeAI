import type { EphemeralProgressImage } from 'features/controlLayers/store/types';
import type { ProgressImage } from 'features/nodes/types/common';
import { round } from 'lodash-es';
import type { MapStore } from 'nanostores';
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

export const useMapSelector = <T extends object>(id: string, map: MapStore<Record<string, T>>): T | undefined => {
  const [value, setValue] = useState<T | undefined>();
  useEffect(() => {
    const unsub = map.subscribe((data) => {
      setValue(data[id]);
    });
    return () => {
      unsub();
    };
  }, [id, map]);

  return value;
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
