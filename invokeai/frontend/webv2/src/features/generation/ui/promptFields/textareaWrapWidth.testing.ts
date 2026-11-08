/**
 * Geometry for the fractional-width mirror tests: a word-prefix of `words` whose one-line width falls inside half a
 * pixel of a textarea padding-box width ending in .5. A mirror sized from the rounded `clientWidth` lands on the other
 * side of that edge in one of the two cases, whichever way the browser rounds.
 */
export type HalfPixelWrapSide = 'fits' | 'wraps';

export interface HalfPixelWrapEdge {
  /** Index in `text` of the word that sits on the wrap edge. */
  edgeWordIndex: number;
  /** The textarea's padding-box width to lay out at: whole pixels plus a half. */
  paddingBoxWidth: number;
  text: string;
}

// Chrome resolves the `font` shorthand to an empty string, so copy the longhands.
const TEXT_STYLES = [
  'fontFamily',
  'fontFeatureSettings',
  'fontKerning',
  'fontSize',
  'fontStretch',
  'fontStyle',
  'fontVariant',
  'fontWeight',
  'letterSpacing',
  'wordSpacing',
] as const;

const measureTextWidth = (computed: CSSStyleDeclaration, text: string): number => {
  const span = document.createElement('span');

  for (const property of TEXT_STYLES) {
    span.style[property] = computed[property];
  }
  span.style.position = 'absolute';
  span.style.whiteSpace = 'pre';
  span.textContent = text;
  document.body.append(span);
  const { width } = span.getBoundingClientRect();
  span.remove();

  return width;
};

export const findHalfPixelWrapEdge = (
  textarea: HTMLTextAreaElement,
  words: readonly string[],
  side: HalfPixelWrapSide
): HalfPixelWrapEdge => {
  const computed = window.getComputedStyle(textarea);
  const paddingInline = Number.parseFloat(computed.paddingLeft) + Number.parseFloat(computed.paddingRight);

  for (let count = 2; count <= words.length; count += 1) {
    const text = words.slice(0, count).join(' ');
    // The narrowest padding box that holds `text` on one line.
    const threshold = measureTextWidth(computed, text) + paddingInline;
    const fraction = threshold - Math.floor(threshold);
    // Stay clear of the edge itself so layout's 1/64 px steps cannot decide the case.
    const isUsable = side === 'wraps' ? fraction >= 0.6 && fraction <= 0.9 : fraction >= 0.1 && fraction <= 0.4;

    if (isUsable) {
      return {
        edgeWordIndex: text.length - words[count - 1]!.length,
        paddingBoxWidth: Math.floor(threshold) + 0.5,
        text,
      };
    }
  }

  throw new Error(`no prefix of the words puts the wrap edge on the ${side} side of a half pixel`);
};
