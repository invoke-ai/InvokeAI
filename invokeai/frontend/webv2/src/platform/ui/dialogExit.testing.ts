import { expect } from 'vitest';

export interface DialogExitFrame {
  state: string | null;
  text: string;
}

/**
 * Close an open dialog and record each `data-state` it passes through, with its text at that moment, until it
 * unmounts or settles closed. A dialog its host unmounted on close never reaches `closed`, so it cannot have animated
 * out; the text shows what it displayed while it did. Independent of animation duration and reduced motion.
 */
export const recordDialogExit = async (
  dialog: Element,
  close: () => Promise<unknown> | unknown
): Promise<DialogExitFrame[]> => {
  const frames: DialogExitFrame[] = [];
  const observer = new MutationObserver(() =>
    frames.push({ state: dialog.getAttribute('data-state'), text: dialog.textContent ?? '' })
  );
  observer.observe(dialog, { attributeFilter: ['data-state'] });

  try {
    await close();
    await expect.poll(() => !dialog.isConnected || dialog.getAttribute('data-state') === 'closed').toBe(true);
  } finally {
    observer.disconnect();
  }

  return frames;
};

/** The frames a dialog showed while closing; empty when its host unmounted it instead. */
export const closingFrames = (frames: readonly DialogExitFrame[]): DialogExitFrame[] =>
  frames.filter((frame) => frame.state === 'closed');
