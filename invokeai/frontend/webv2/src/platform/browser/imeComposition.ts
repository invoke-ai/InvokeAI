/**
 * Whether a keydown belongs to an IME: while it composes, arrows and Enter choose and confirm candidates, so a
 * handler must leave them alone. Safari dispatches the confirming keydown after `compositionend`, which leaves
 * keyCode 229 as the only sign.
 */
export const isImeComposing = (event: Pick<KeyboardEvent, 'isComposing' | 'keyCode'>): boolean =>
  event.isComposing || event.keyCode === 229;
