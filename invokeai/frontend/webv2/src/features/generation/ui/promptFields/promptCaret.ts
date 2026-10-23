export interface CaretRect {
  height: number;
  width: number;
  x: number;
  y: number;
}

/**
 * The width a textarea wraps its text in: its padding box minus any vertical scrollbar, which is what `clientWidth`
 * reports, but without `clientWidth`'s rounding to whole pixels. At a fractional layout width a mirror sized from the
 * rounded value is up to half a pixel wider than the textarea and can keep a word on a line the textarea wraps.
 *
 * Pass the textarea's live computed style when measuring repeatedly, so it is looked up once.
 */
export const getTextareaWrapWidth = (
  textarea: HTMLTextAreaElement,
  computed: CSSStyleDeclaration = window.getComputedStyle(textarea)
): number => {
  const width = Number.parseFloat(computed.width);

  if (!Number.isFinite(width)) {
    return textarea.clientWidth;
  }

  const borderInline =
    (Number.parseFloat(computed.borderLeftWidth) || 0) + (Number.parseFloat(computed.borderRightWidth) || 0);
  const paddingBoxWidth =
    computed.boxSizing === 'border-box'
      ? width - borderInline
      : width + (Number.parseFloat(computed.paddingLeft) || 0) + (Number.parseFloat(computed.paddingRight) || 0);
  // A scrollbar takes whole pixels, so the rounded box sizes measure it exactly; overlay scrollbars measure 0.
  const scrollbar = Math.max(0, Math.round(textarea.offsetWidth - textarea.clientWidth - borderInline));

  return paddingBoxWidth - scrollbar;
};

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

// Reuse the mirror to avoid DOM insertion/removal on each autocomplete keystroke.
let mirrorElements: { marker: HTMLSpanElement; mirror: HTMLDivElement } | null = null;

const getMirror = (): { marker: HTMLSpanElement; mirror: HTMLDivElement } => {
  if (!mirrorElements) {
    const mirror = document.createElement('div');
    const marker = document.createElement('span');

    mirror.setAttribute('aria-hidden', 'true');
    mirror.append(marker);
    document.body.append(mirror);
    mirrorElements = { marker, mirror };
  }

  return mirrorElements;
};

export const getTextareaCaretRect = (textarea: HTMLTextAreaElement, index: number): CaretRect | null => {
  const textareaRect = textarea.getBoundingClientRect();

  if (textareaRect.width === 0 && textareaRect.height === 0) {
    return null;
  }

  const computed = window.getComputedStyle(textarea);
  const { marker, mirror } = getMirror();

  for (const property of MIRRORED_STYLES) {
    mirror.style[property] = computed[property];
  }

  mirror.style.position = 'absolute';
  mirror.style.top = '0';
  mirror.style.left = '-9999px';
  mirror.style.visibility = 'hidden';
  mirror.style.whiteSpace = 'pre-wrap';
  mirror.style.overflowWrap = 'break-word';
  mirror.style.overflow = 'hidden';
  mirror.style.boxSizing = 'border-box';
  mirror.style.border = '0';
  mirror.style.width = `${getTextareaWrapWidth(textarea, computed)}px`;

  mirror.textContent = textarea.value.slice(0, index);
  marker.textContent = textarea.value.slice(index) || '​';
  mirror.append(marker);

  const lineHeight = Number.parseFloat(computed.lineHeight);
  const borderLeft = Number.parseFloat(computed.borderLeftWidth) || 0;
  const borderTop = Number.parseFloat(computed.borderTopWidth) || 0;
  const rect: CaretRect = {
    height: Number.isFinite(lineHeight) ? lineHeight : Number.parseFloat(computed.fontSize) * 1.6,
    width: 0,
    x: textareaRect.left + borderLeft + marker.offsetLeft - textarea.scrollLeft,
    y: textareaRect.top + borderTop + marker.offsetTop - textarea.scrollTop,
  };

  mirror.textContent = '';

  return rect;
};
