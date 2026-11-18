import { describe, expect, it } from 'vitest';

import type { ServerOutputStyleRun } from './transcript';

import { extendedPaletteHex, parseOutputColor, segmentRow } from './rowSegments';

const segments = (text: string, styles: ServerOutputStyleRun[]) =>
  segmentRow({ id: 1, styles, text, truncated: false }).map(({ style, text: part }) => [part, style]);

const plain = { bold: false, dim: false, italic: false, underline: false };

describe('segmentRow', () => {
  it('splits text at style runs, leaving gaps unstyled', () => {
    expect(
      segments('[INFO] ready', [
        { fg: '6', length: 4, offset: 1 },
        { bold: true, length: 5, offset: 7 },
      ])
    ).toEqual([
      ['[', null],
      ['INFO', { ...plain, bg: undefined, fg: { index: 6, kind: 'palette' } }],
      ['] ', null],
      ['ready', { ...plain, bg: undefined, bold: true, fg: undefined }],
    ]);
  });

  it('keeps every character once when runs overlap, run backwards or pass the end', () => {
    expect(
      segments('abcdef', [
        { length: 3, offset: 1, underline: true },
        { length: 2, offset: 2, italic: true },
        { length: 99, offset: 5, dim: true },
        { length: 1, offset: 40, bold: true },
      ]).map(([text]) => text)
    ).toEqual(['a', 'bcd', 'e', 'f']);
  });

  it('swaps colors for inverse, standing in the view defaults for missing ones', () => {
    expect(segments('x', [{ fg: '#AABBCC', inverse: true, length: 1, offset: 0 }])[0]![1]).toMatchObject({
      bg: { hex: '#aabbcc', kind: 'rgb' },
      fg: { kind: 'default' },
    });
  });

  it('renders an empty row as one empty segment', () => {
    expect(segments('', [])).toEqual([['', null]]);
  });
});

describe('parseOutputColor', () => {
  it.each(['0', '15', '255', '#000000', '#a1B2c3'])('accepts %s', (value) => {
    expect(parseOutputColor(value)).toBeDefined();
  });

  it.each(['256', '-1', '01', 'red', '#fff', '#12345g', 'url(x)', '1;color:red', '#123456;x', 7, null])(
    'drops %s',
    (value) => {
      expect(parseOutputColor(value)).toBeUndefined();
    }
  );

  it('maps the extended palette to the xterm table', () => {
    expect([16, 21, 196, 231, 232, 255].map(extendedPaletteHex)).toEqual([
      '#000000',
      '#0000ff',
      '#ff0000',
      '#ffffff',
      '#080808',
      '#eeeeee',
    ]);
  });
});
