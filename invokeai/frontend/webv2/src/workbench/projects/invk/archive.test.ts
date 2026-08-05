import { describe, expect, it } from 'vitest';

import {
  binaryEntry,
  createExpansionBudget,
  INVK_MAX_ENTRIES,
  readArchive,
  readEntryText,
  textEntry,
  writeArchive,
} from './archive';
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

  it('refuses to write more entries than the ceiling allows', async () => {
    const tooMany = new Map(
      Array.from({ length: INVK_MAX_ENTRIES + 1 }, (_, index) => [
        `images/${index}.png`,
        binaryEntry(new Uint8Array([1])),
      ])
    );

    await expect(writeArchive(tooMany)).rejects.toMatchObject({ reason: 'too-large' });
  });
});

/**
 * The point of this guard is *when* it fires. `unzip` is fully buffered, so a
 * total measured over its result describes a bomb that has already landed; the
 * budget is what fflate consults per entry before inflating it. Testing it
 * directly is how the sizes involved stay expressible — proving the same thing
 * through `readArchive` would mean actually building a multi-gigabyte archive.
 */
describe('createExpansionBudget', () => {
  const entry = (name: string, originalSize: number, size = originalSize, compression = 8) => ({
    compression,
    name,
    originalSize,
    size,
  });
  const GIB = 1024 * 1024 * 1024;

  it('accepts entries that fit and raises nothing', () => {
    const budget = createExpansionBudget();

    expect(budget.accept(entry('project.json', 1024))).toBe(true);
    expect(budget.accept(entry('images/a.png', 4096))).toBe(true);
    expect(budget.getRefusal()).toBeNull();
  });

  it('refuses the entry that crosses the byte ceiling, and every entry after it', () => {
    const budget = createExpansionBudget();

    expect(budget.accept(entry('images/a.png', GIB))).toBe(true);
    // 2 GiB total is the ceiling itself, which still fits.
    expect(budget.accept(entry('images/b.png', GIB))).toBe(true);
    expect(budget.accept(entry('images/c.png', 1))).toBe(false);
    expect(budget.accept(entry('images/d.png', 1))).toBe(false);

    expect(budget.getRefusal()).toBeInstanceOf(InvkFormatError);
    expect(budget.getRefusal()).toMatchObject({ reason: 'too-large' });
  });

  it('refuses past the entry ceiling before any of the excess is inflated', () => {
    const budget = createExpansionBudget();

    for (let index = 0; index < INVK_MAX_ENTRIES; index += 1) {
      expect(budget.accept(entry(`images/${index}.png`, 1))).toBe(true);
    }

    expect(budget.accept(entry('images/one-too-many.png', 1))).toBe(false);
    expect(budget.getRefusal()).toMatchObject({ reason: 'too-large' });
  });

  it('skips directory records without spending budget on them', () => {
    const budget = createExpansionBudget();

    for (let index = 0; index <= INVK_MAX_ENTRIES; index += 1) {
      expect(budget.accept(entry(`images/${index}/`, 0))).toBe(false);
    }

    expect(budget.getRefusal()).toBeNull();
    expect(budget.accept(entry('project.json', 1))).toBe(true);
  });

  it('keeps the first reason when both ceilings are crossed', () => {
    const budget = createExpansionBudget();

    budget.accept(entry('images/huge.png', 3 * GIB));
    const first = budget.getRefusal();

    budget.accept(entry('images/also-huge.png', 3 * GIB));

    expect(budget.getRefusal()).toBe(first);
  });

  it('refuses a stored entry whose compressed and original sizes disagree', () => {
    const budget = createExpansionBudget();

    expect(budget.accept(entry('images/tampered.png', 1, 1024 * 1024, 0))).toBe(false);
    expect(budget.getRefusal()).toMatchObject({ reason: 'not-a-project' });
  });
});

describe('readArchive with the budget in the path', () => {
  it('round-trips an ordinary archive, so the filter is permissive at real sizes', async () => {
    const blob = await writeArchive(
      new Map([
        ['project.json', textEntry('a'.repeat(4096))],
        ['images/a.png', binaryEntry(new Uint8Array(4096))],
      ])
    );

    const entries = await readArchive(new Uint8Array(await blob.arrayBuffer()));

    expect([...entries.keys()].sort()).toEqual(['images/a.png', 'project.json']);
    expect(entries.get('images/a.png')!.byteLength).toBe(4096);
  });

  it('rejects a stored payload whose central directory understates its allocation', async () => {
    const blob = await writeArchive(
      new Map([['images/tampered.png', binaryEntry(new Uint8Array(1024 * 1024).fill(0xa5))]])
    );
    const bytes = new Uint8Array(await blob.arrayBuffer());
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let centralDirectoryOffset = -1;

    for (let offset = 0; offset <= bytes.byteLength - 4; offset += 1) {
      if (view.getUint32(offset, true) === 0x02014b50) {
        centralDirectoryOffset = offset;
        break;
      }
    }

    expect(centralDirectoryOffset).toBeGreaterThanOrEqual(0);
    // ZIP central-directory uncompressed size. The stored size remains the real 1 MiB.
    view.setUint32(centralDirectoryOffset + 24, 1, true);

    await expect(readArchive(bytes)).rejects.toMatchObject({ reason: 'not-a-project' });
  });
});
