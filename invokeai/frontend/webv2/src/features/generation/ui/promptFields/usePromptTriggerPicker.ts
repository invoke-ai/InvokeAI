/** Browsing takes focus; autocomplete does not. Anchor both to the supplied rectangle. */

import type { PromptTextRange } from '@features/generation/ui/promptFields/promptFocus';
import type { ReactNode } from 'react';

import { DismissOnViewportChange } from '@features/generation/ui/promptFields/useDismissOnViewportChange';
import { createElement, useCallback, useMemo, useState } from 'react';

interface AnchorRect {
  height: number;
  width: number;
  x: number;
  y: number;
}

export interface PromptTriggerPickerApi {
  isOpen: boolean;
  positioning: { getAnchorRect: () => AnchorRect | null };
  close: () => void;
  dismissElement: ReactNode;
  open: (anchorElement: HTMLElement) => void;
  /** Inserts the picked trigger, then closes. */
  select: (trigger: string) => void;
}

export const usePromptTriggerPicker = ({
  insert,
}: {
  insert: (trigger: string, range?: PromptTextRange) => void;
}): PromptTriggerPickerApi => {
  const [anchorRect, setAnchorRect] = useState<AnchorRect | null>(null);

  const open = useCallback((anchorElement: HTMLElement) => {
    const rect = anchorElement.getBoundingClientRect();

    setAnchorRect({ height: rect.height, width: rect.width, x: rect.x, y: rect.y });
  }, []);

  const close = useCallback(() => setAnchorRect(null), []);

  const select = useCallback(
    (trigger: string) => {
      insert(trigger);
      setAnchorRect(null);
    },
    [insert]
  );

  const positioning = useMemo(() => ({ getAnchorRect: () => anchorRect }), [anchorRect]);

  const dismissElement =
    anchorRect === null ? null : createElement(DismissOnViewportChange, { dismiss: close, enabled: true });

  return { close, dismissElement, isOpen: anchorRect !== null, open, positioning, select };
};
