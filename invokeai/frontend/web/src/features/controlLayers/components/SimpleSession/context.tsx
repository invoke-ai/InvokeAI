import type {
  AdvancedSessionIdentifier,
  SimpleSessionIdentifier,
} from 'features/controlLayers/store/canvasStagingAreaSlice';
import type { ProgressImage } from 'features/nodes/types/common';
import { atom, type WritableAtom } from 'nanostores';
import type { PropsWithChildren } from 'react';
import { createContext, memo, useContext, useEffect, useState } from 'react';
import type { S } from 'services/api/types';
import { assert } from 'tsafe';

export type ProgressData = {
  sessionId: string;
  progressEvent: S['InvocationProgressEvent'] | null;
  progressImage: ProgressImage | null;
};

export const buildProgressDataAtom = () => atom<Record<string, ProgressData>>({});

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

export type CanvasSessionContextValue = {
  session: SimpleSessionIdentifier | AdvancedSessionIdentifier;
  $progressData: WritableAtom<Record<string, ProgressData>>;
};

const CanvasSessionContext = createContext<CanvasSessionContextValue | null>(null);

export const CanvasSessionContextProvider = memo(
  ({ value, children }: PropsWithChildren<{ value: CanvasSessionContextValue }>) => (
    <CanvasSessionContext.Provider value={value}>{children}</CanvasSessionContext.Provider>
  )
);
CanvasSessionContextProvider.displayName = 'CanvasSessionContextProvider';

export const useCanvasSessionContext = () => {
  const ctx = useContext(CanvasSessionContext);
  assert(ctx !== null, "'useCanvasSessionContext' must be used within a CanvasSessionContextProvider");
  return ctx;
};
