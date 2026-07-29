export interface CaretRect {
  height: number;
  width: number;
  x: number;
  y: number;
}

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
  mirror.style.width = `${textarea.clientWidth}px`;

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
