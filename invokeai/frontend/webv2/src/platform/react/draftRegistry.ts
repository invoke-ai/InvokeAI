import { useMountEffect } from '@platform/react/useMountEffect';
import { useEffectEvent } from 'react';

type DraftFlusher = () => void;

const draftFlushers = new Set<DraftFlusher>();

/** Commits pending drafts from every mounted widget before a project snapshot boundary. */
export const flushWorkbenchDrafts = (): void => {
  for (const flushDraft of Array.from(draftFlushers)) {
    flushDraft();
  }
};

export const useRegisterDraftFlusher = (flushDraft: DraftFlusher): void => {
  const flushLatestDraft = useEffectEvent(flushDraft);

  useMountEffect(() => {
    const registeredFlushDraft = () => flushLatestDraft();

    draftFlushers.add(registeredFlushDraft);

    return () => {
      draftFlushers.delete(registeredFlushDraft);
    };
  });
};
