import { describe, expect, it } from 'vitest';

import { binaryEntry, readArchive, readEntryText, textEntry, writeArchive } from './archive';
import { InvkFormatError } from './format';

const toBytes = async (blob: Blob): Promise<Uint8Array> => new Uint8Array(await blob.arrayBuffer());

describe('writeArchive / readArchive', () => {
  it('round-trips text and binary entries', async () => {
    const pixels = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0, 1, 2, 3, 255]);
    const blob = await writeArchive(
      new Map([
        ['manifest.json', textEntry('{"version":2}')],
        ['images/a.png', binaryEntry(pixels)],
      ])
    );

    const entries = await readArchive(await toBytes(blob));

    expect([...entries.keys()].sort()).toEqual(['images/a.png', 'manifest.json']);
    expect(readEntryText(entries.get('manifest.json')!)).toBe('{"version":2}');
    expect([...entries.get('images/a.png')!]).toEqual([...pixels]);
  });

  it('round-trips entries from both asset folders', async () => {
    const blob = await writeArchive(
      new Map([
        ['images/a.png', binaryEntry(new Uint8Array([1, 2]))],
        ['videos/clip.mp4', binaryEntry(new Uint8Array([3, 4, 5]))],
      ])
    );

    const entries = await readArchive(await toBytes(blob));

    expect([...entries.keys()].sort()).toEqual(['images/a.png', 'videos/clip.mp4']);
    expect([...entries.get('videos/clip.mp4')!]).toEqual([3, 4, 5]);
  });

  it('preserves unicode in text entries', async () => {
    const blob = await writeArchive(new Map([['project.json', textEntry('{"name":"プロジェクト — 1"}')]]));
    const entries = await readArchive(await toBytes(blob));

    expect(readEntryText(entries.get('project.json')!)).toBe('{"name":"プロジェクト — 1"}');
  });

  it('actually compresses repetitive text', async () => {
    const repetitive = JSON.stringify({
      layers: Array.from({ length: 500 }, (_, index) => ({ id: `layer-${index}` })),
    });
    const blob = await writeArchive(new Map([['project.json', textEntry(repetitive)]]));

    expect(blob.size).toBeLessThan(repetitive.length / 2);
  });

  it('drops directory records', async () => {
    const blob = await writeArchive(
      new Map([
        ['images/a.png', binaryEntry(new Uint8Array([1]))],
        ['images/b.png', binaryEntry(new Uint8Array([2]))],
      ])
    );

    const entries = await readArchive(await toBytes(blob));

    expect([...entries.keys()].sort()).toEqual(['images/a.png', 'images/b.png']);
  });

  it('reports a non-ZIP as not a project', async () => {
    await expect(readArchive(new TextEncoder().encode('this is not a zip'))).rejects.toMatchObject({
      reason: 'not-a-project',
    });
  });

  it('reports an oversized input rather than expanding it', async () => {
    const oversized = { byteLength: 5 * 1024 * 1024 * 1024 } as Uint8Array;

    await expect(readArchive(oversized)).rejects.toBeInstanceOf(InvkFormatError);
    await expect(readArchive(oversized)).rejects.toMatchObject({ reason: 'too-large' });
  });

  it('refuses to write past the archive ceiling', async () => {
    const oversized = new Map([['images/huge.png', { bytes: { byteLength: 5 * 1024 * 1024 * 1024 }, kind: 'binary' }]]);

    await expect(writeArchive(oversized as never)).rejects.toMatchObject({ reason: 'too-large' });
  });
});
