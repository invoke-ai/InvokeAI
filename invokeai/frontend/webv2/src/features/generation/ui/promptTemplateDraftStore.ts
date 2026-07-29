/**
 * A one-shot handoff for "save this image's prompt as a template".
 *
 * The action starts in the gallery's context menu but has to finish in the
 * generate widget's template editor, which is inside a popover the menu cannot
 * reach. Rather than lifting the editor into a global dialog, the menu leaves a
 * draft here and opens the widget; the templates button picks it up and opens
 * the editor prefilled.
 *
 * The draft is consumed on read, so re-opening the popover later starts on the
 * list rather than resurrecting an old handoff.
 */

import { useMountEffect } from '@platform/react/useMountEffect';
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

export const setPendingPromptTemplateDraft = (draft: PendingPromptTemplateDraft): void => store.setSnapshot({ draft });

const takePendingPromptTemplateDraft = (): PendingPromptTemplateDraft | null => {
  const { draft } = store.getSnapshot();

  if (draft) {
    store.setSnapshot({ draft: null });
  }

  return draft;
};

/**
 * Delivers a handed-over draft to the editor exactly once — on subscribe for one
 * left before the widget mounted, and on notification thereafter.
 *
 * Deliberately a subscription with a callback rather than a rendered value: the
 * draft is a one-shot event from outside React, and reading it during render
 * would mean clearing it in an effect, which is the cascading-render pattern the
 * compiler rejects.
 */
export const useOnPendingPromptTemplateDraft = (onDraft: (draft: PendingPromptTemplateDraft) => void): void => {
  const deliver = useEffectEvent(() => {
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
