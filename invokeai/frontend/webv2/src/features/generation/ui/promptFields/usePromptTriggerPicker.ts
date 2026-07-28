/**
 * The `+` button's browsable trigger popover.
 *
 * Distinct from the caret autocomplete beside it: this one is opened
 * deliberately, takes focus, and browses every trigger there is. Both prompt
 * fields wire it up identically, so it lives here rather than twice.
 *
 * The popover is positioned by rect rather than by element so it can anchor to
 * the button that opened it, wherever that sits, without the field having to
 * hand its own DOM to a child.
 */

import type { PromptTextRange } from '@features/generation/ui/promptFields/promptFocus';

import { useCallback, useMemo, useState } from 'react';

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

  return { close, isOpen: anchorRect !== null, open, positioning, select };
};
