/**
 * Where the caret is on screen, so a list can open beside it.
 *
 * A textarea gives no geometry for a position within its text, so the standard
 * trick is to lay the same text out again in a hidden element that copies every
 * property affecting how it wraps, and read the offset of a marker placed at the
 * caret. The prompt textarea already renders a mirror for syntax highlighting,
 * but only when highlighting is on — this builds its own so caret anchoring
 * works with the setting off too.
 *
 * Measured on demand rather than tracked: it is read while a picker is opening
 * or its query is growing, which is a handful of times per keystroke at most,
 * and never while the user is only typing.
 */

export interface CaretRect {
  height: number;
  width: number;
  x: number;
  y: number;
}

/**
 * Everything that decides where a character lands, copied onto the mirror.
 *
 * Width is deliberately absent, and set from `clientWidth` below instead.
 * Copying the resolved `width` does work — it is the border-box width under
 * `box-sizing: border-box`, which the copied borders and padding then reduce to
 * the same content width — but it only works because those three travel
 * together, and it counts a vertical scrollbar as space for text. `clientWidth`
 * is the padding box with the scrollbar already taken out, which is the width
 * the field actually wraps at, stated once.
 */
const MIRRORED_STYLES = [
  'direction',
  'fontFamily',
  'fontSize',
  'fontStretch',
  'fontStyle',
  'fontVariant',
  'fontWeight',
  'letterSpacing',
  'lineHeight',
  'paddingBottom',
  'paddingLeft',
  'paddingRight',
  'paddingTop',
  'tabSize',
  'textAlign',
  'textIndent',
  'textTransform',
  'wordSpacing',
] as const;

/**
 * The caret's viewport rectangle, or `null` when the element is not laid out.
 *
 * The returned box is a zero-width line at the caret: callers place a popover
 * against it, so its height is what decides whether the list sits above or below
 * the current line rather than on top of it.
 */
export const getTextareaCaretRect = (textarea: HTMLTextAreaElement, index: number): CaretRect | null => {
  const textareaRect = textarea.getBoundingClientRect();

  if (textareaRect.width === 0 && textareaRect.height === 0) {
    return null;
  }

  const computed = window.getComputedStyle(textarea);
  const mirror = document.createElement('div');
  const marker = document.createElement('span');

  for (const property of MIRRORED_STYLES) {
    mirror.style[property] = computed[property];
  }

  // Off-screen rather than `display: none`, which would give every offset zero.
  mirror.style.position = 'absolute';
  mirror.style.top = '0';
  mirror.style.left = '-9999px';
  mirror.style.visibility = 'hidden';
  mirror.style.whiteSpace = 'pre-wrap';
  mirror.style.overflowWrap = 'break-word';
  mirror.style.overflow = 'hidden';
  // `clientWidth` is the padding box, so with no border of its own the mirror
  // gets exactly the field's own wrapping width.
  mirror.style.boxSizing = 'border-box';
  mirror.style.border = '0';
  mirror.style.width = `${textarea.clientWidth}px`;

  mirror.textContent = textarea.value.slice(0, index);
  // A zero-width space so the marker still occupies a line when the caret sits
  // at the very end of the text, or straight after a newline.
  marker.textContent = textarea.value.slice(index) || '​';
  mirror.append(marker);
  document.body.append(mirror);

  const lineHeight = Number.parseFloat(computed.lineHeight);
  // The offsets are measured from the mirror's padding edge, which the field's
  // own border sits outside of, so it has to be added back by hand.
  const borderLeft = Number.parseFloat(computed.borderLeftWidth) || 0;
  const borderTop = Number.parseFloat(computed.borderTopWidth) || 0;
  const rect: CaretRect = {
    height: Number.isFinite(lineHeight) ? lineHeight : Number.parseFloat(computed.fontSize) * 1.6,
    width: 0,
    x: textareaRect.left + borderLeft + marker.offsetLeft - textarea.scrollLeft,
    y: textareaRect.top + borderTop + marker.offsetTop - textarea.scrollTop,
  };

  mirror.remove();

  return rect;
};
