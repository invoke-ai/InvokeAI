import { downloadBlob, downloadText } from '@platform/browser/downloadBlob';
import { afterEach, describe, expect, it, vi } from 'vitest';

/** Record connectivity at click time; the helper immediately detaches the anchor afterward. */
const capture = () => {
  const clicked: { download: string; href: string; wasConnected: boolean }[] = [];

  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function mockClick(this: HTMLAnchorElement) {
    clicked.push({ download: this.download, href: this.href, wasConnected: this.isConnected });
  });

  return { clicked };
};

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('downloadBlob', () => {
  it('names the download and points it at the blob', () => {
    const { clicked } = capture();

    downloadBlob(new Blob(['hi'], { type: 'text/plain' }), 'notes.txt');

    expect(clicked).toHaveLength(1);
    expect(clicked[0]!.download).toBe('notes.txt');
    expect(clicked[0]!.href.startsWith('blob:')).toBe(true);
  });

  // Attach before click for Firefox compatibility.
  it('attaches the anchor to the document before clicking it', () => {
    const { clicked } = capture();

    downloadBlob(new Blob(['hi']), 'notes.txt');

    expect(clicked[0]!.wasConnected).toBe(true);
  });

  it('leaves nothing behind in the document', () => {
    capture();

    downloadBlob(new Blob(['hi']), 'notes.txt');

    expect(document.querySelector('a[download="notes.txt"]')).toBeNull();
  });

  // Defer revocation to avoid racing Safari's URL consumption.
  it('keeps the object URL alive past the click', async () => {
    const { clicked } = capture();
    const revoke = vi.spyOn(URL, 'revokeObjectURL');

    downloadBlob(new Blob(['hi']), 'notes.txt');

    expect(revoke).not.toHaveBeenCalled();

    await new Promise((resolve) => {
      requestAnimationFrame(() => requestAnimationFrame(resolve));
    });

    expect(revoke).toHaveBeenCalledWith(clicked[0]!.href);
  });

  it('wraps text with its content type', async () => {
    const { clicked } = capture();

    downloadText('a,b\n1,2', 'table.csv', 'text/csv');

    expect(clicked[0]!.download).toBe('table.csv');
    await expect(fetch(clicked[0]!.href).then((response) => response.text())).resolves.toBe('a,b\n1,2');
  });
});

it('settles owned cleanup when a background tab never delivers animation frames', async () => {
  vi.useFakeTimers();
  capture();
  vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(123);
  const cancel = vi.spyOn(window, 'cancelAnimationFrame');
  const revoke = vi.spyOn(URL, 'revokeObjectURL');
  const pending = downloadBlob(new Blob(['background']), 'background.txt');
  expect(revoke).not.toHaveBeenCalled();
  await vi.advanceTimersByTimeAsync(1000);
  await pending;
  expect(revoke).toHaveBeenCalledOnce();
  expect(cancel).toHaveBeenCalledWith(123);
});

it('releases its anchor and URL immediately when the browser rejects the click', () => {
  capture();
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {
    throw new Error('click failed');
  });
  const revoke = vi.spyOn(URL, 'revokeObjectURL');
  expect(() => downloadBlob(new Blob(['failure']), 'failed.txt')).toThrow('click failed');
  expect(revoke).toHaveBeenCalledOnce();
  expect(document.querySelector('a[download="failed.txt"]')).toBeNull();
});
