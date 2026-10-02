import { expect, it, vi } from 'vitest';

import { INVK_MAX_ARCHIVE_BYTES, textEntry, writeArchive } from './archive';

vi.mock('fflate', () => ({
  zip: (_entries: unknown, callback: (error: Error | null, data: Uint8Array) => void) =>
    callback(null, { byteLength: INVK_MAX_ARCHIVE_BYTES + 1 } as Uint8Array),
}));

it('refuses oversized packed bytes before constructing a Blob', async () => {
  const blob = vi.spyOn(globalThis, 'Blob');

  await expect(writeArchive(new Map([['project.json', textEntry('{}')]]))).rejects.toMatchObject({
    reason: 'too-large',
  });
  expect(blob).not.toHaveBeenCalled();
});
