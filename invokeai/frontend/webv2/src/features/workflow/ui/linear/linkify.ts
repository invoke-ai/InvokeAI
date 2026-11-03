export type TextSegment = { kind: 'text'; value: string } | { kind: 'link'; value: string };

const URL_PATTERN = /https?:\/\/[^\s<>"'`]+/g;
// Sentence punctuation that follows a URL in prose is not part of it.
const TRAILING_PUNCTUATION = /[.,;:!?]+$/;

const trimUrl = (raw: string): string => {
  let url = raw.replace(TRAILING_PUNCTUATION, '');

  // A closing bracket belongs to the URL only when the URL opened it ("(see https://x/y(z))" vs. "(https://x)").
  while (url.endsWith(')') && (url.match(/\(/g) ?? []).length < (url.match(/\)/g) ?? []).length) {
    url = url.slice(0, -1).replace(TRAILING_PUNCTUATION, '');
  }

  return url;
};

/** Splits prose into literal text and `http(s)` URLs, in order, so a renderer can turn the URLs into anchors. */
export const splitTextIntoLinkSegments = (text: string): TextSegment[] => {
  const segments: TextSegment[] = [];
  let cursor = 0;

  for (const match of text.matchAll(URL_PATTERN)) {
    const url = trimUrl(match[0]);

    if (url.length === 0) {
      continue;
    }

    if (match.index > cursor) {
      segments.push({ kind: 'text', value: text.slice(cursor, match.index) });
    }

    segments.push({ kind: 'link', value: url });
    cursor = match.index + url.length;
  }

  if (cursor < text.length) {
    segments.push({ kind: 'text', value: text.slice(cursor) });
  }

  return segments;
};
