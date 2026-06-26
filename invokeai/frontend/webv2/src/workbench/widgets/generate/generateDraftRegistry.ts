import { useEffect, useEffectEvent } from 'react';

type DraftFlusher = () => void;

const draftFlushers = new Set<DraftFlusher>();

export const flushGenerateDrafts = (): void => {
  const flushers = Array.from(draftFlushers);

  for (const flushDraft of flushers) {
    flushDraft();
  }
};

export const useRegisterGenerateDraftFlusher = (flushDraft: DraftFlusher): void => {
  const flushLatestDraft = useEffectEvent(flushDraft);

  useEffect(() => {
    const registeredFlushDraft = () => flushLatestDraft();

    draftFlushers.add(registeredFlushDraft);

    return () => {
      draftFlushers.delete(registeredFlushDraft);
    };
  }, []);
};
