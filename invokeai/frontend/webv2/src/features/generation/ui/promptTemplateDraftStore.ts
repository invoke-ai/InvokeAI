import { useMountEffect } from '@platform/react/useMountEffect';
import { registerAccountOwnedResource } from '@platform/state/accountLifecycle';
import { createExternalStore } from '@platform/state/externalStore';
import { useEffectEvent } from 'react';

export interface PendingPromptTemplateDraft {
  negativePrompt: string;
  positivePrompt: string;
}

interface PendingPromptTemplateDraftState {
  draft: PendingPromptTemplateDraft | null;
}

const store = createExternalStore<PendingPromptTemplateDraftState>({ draft: null });

registerAccountOwnedResource({
  clear: () => store.setSnapshot({ draft: null }),
  name: 'pending-prompt-template-draft',
});

export const setPendingPromptTemplateDraft = (draft: PendingPromptTemplateDraft): void => store.setSnapshot({ draft });

const takePendingPromptTemplateDraft = (): PendingPromptTemplateDraft | null => {
  const { draft } = store.getSnapshot();

  if (draft) {
    store.setSnapshot({ draft: null });
  }

  return draft;
};

/** `isEnabled` false leaves a draft for another field: only the surface the handover reveals takes it. */
export const useOnPendingPromptTemplateDraft = (
  onDraft: (draft: PendingPromptTemplateDraft) => void,
  isEnabled = true
): void => {
  const deliver = useEffectEvent(() => {
    if (!isEnabled) {
      return;
    }

    const draft = takePendingPromptTemplateDraft();

    if (draft) {
      onDraft(draft);
    }
  });

  /* eslint-disable react-hooks/rules-of-hooks -- useMountEffect is the repository's explicit useEffect wrapper */
  useMountEffect(() => {
    deliver();
    return store.subscribe(deliver);
  });
  /* eslint-enable react-hooks/rules-of-hooks */
};
