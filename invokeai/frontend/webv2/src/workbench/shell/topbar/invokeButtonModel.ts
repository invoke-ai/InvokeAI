export type InvokeIconMode = { mode: 'play' } | { mode: 'progress'; value: number | null };

/**
 * The Invoke button's icon slot. Progress replaces the play glyph only while a
 * batch runs AND the pointer is elsewhere; hovering always restores the play
 * glyph so "queue more on top" stays legible. (Contract §9.4: geometry, width,
 * and enabled state never change — only the icon slot's content may.)
 */
export const getInvokeIconMode = ({
  hasOpenWork,
  isHovered,
  progress,
}: {
  hasOpenWork: boolean;
  isHovered: boolean;
  progress: number | null;
}): InvokeIconMode => (hasOpenWork && !isHovered ? { mode: 'progress', value: progress } : { mode: 'play' });
