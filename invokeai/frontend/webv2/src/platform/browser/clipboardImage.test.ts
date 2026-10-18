import { describe, expect, it } from 'vitest';

import { readClipboardImage } from './clipboardImage';

const blob = (type: string): Blob => ({ type }) as Blob;

/** A ClipboardItem stand-in exposing just what the reader touches. */
const item = (types: string[], payload: Record<string, Blob> = {}): ClipboardItem =>
  ({
    getType: (type: string) => Promise.resolve(payload[type] ?? blob(type)),
    types,
  }) as unknown as ClipboardItem;

describe('readClipboardImage', () => {
  it('returns null when the clipboard API is unavailable', async () => {
    await expect(readClipboardImage({ clipboard: undefined })).resolves.toBeNull();
  });

  it('returns null when permission is refused rather than throwing', async () => {
    // A paste that cannot read the clipboard is an ordinary outcome — a
    // defocused document refuses too — so it must not surface as an error.
    const clipboard = { read: () => Promise.reject(new Error('NotAllowedError')) };
    await expect(readClipboardImage({ clipboard })).resolves.toBeNull();
  });

  it('returns null when the clipboard holds no image', async () => {
    const clipboard = { read: () => Promise.resolve([item(['text/plain'])]) };
    await expect(readClipboardImage({ clipboard })).resolves.toBeNull();
  });

  it('reads a PNG off the clipboard', async () => {
    const png = blob('image/png');
    const clipboard = { read: () => Promise.resolve([item(['image/png'], { 'image/png': png })]) };
    await expect(readClipboardImage({ clipboard })).resolves.toBe(png);
  });

  it('prefers PNG over lossy types when several are offered', async () => {
    const png = blob('image/png');
    const jpeg = blob('image/jpeg');
    const clipboard = {
      read: () => Promise.resolve([item(['image/jpeg', 'image/png'], { 'image/jpeg': jpeg, 'image/png': png })]),
    };
    await expect(readClipboardImage({ clipboard })).resolves.toBe(png);
  });

  it('falls back to a lossy type when PNG is absent', async () => {
    const jpeg = blob('image/jpeg');
    const clipboard = { read: () => Promise.resolve([item(['image/jpeg'], { 'image/jpeg': jpeg })]) };
    await expect(readClipboardImage({ clipboard })).resolves.toBe(jpeg);
  });

  it('scans past a non-image item to a later image', async () => {
    const png = blob('image/png');
    const clipboard = {
      read: () => Promise.resolve([item(['text/plain']), item(['image/png'], { 'image/png': png })]),
    };
    await expect(readClipboardImage({ clipboard })).resolves.toBe(png);
  });
});
