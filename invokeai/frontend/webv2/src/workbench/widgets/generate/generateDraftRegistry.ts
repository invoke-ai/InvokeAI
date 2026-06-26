import { useEffect, useRef } from 'react';

type DraftFlusher = () => void;

const draftFlushers = new Set<DraftFlusher>();

export const flushGenerateDrafts = (): void => {
  const flushers = Array.from(draftFlushers);

  for (const flushDraft of flushers) {
    flushDraft();
  }
};

export const useRegisterGenerateDraftFlusher = (flushDraft: DraftFlusher): void => {
  const flushDraftRef = useRef(flushDraft);

  flushDraftRef.current = flushDraft;

  useEffect(() => {
    const registeredFlushDraft = () => flushDraftRef.current();

    draftFlushers.add(registeredFlushDraft);

    return () => {
      draftFlushers.delete(registeredFlushDraft);
    };
  }, []);
};
