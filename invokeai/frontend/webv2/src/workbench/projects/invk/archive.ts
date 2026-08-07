import { INVK_MIME_TYPE, InvkFormatError } from './format';

/**
 * The ZIP seam. The only module that knows `.invk` is a ZIP, and the only one that imports fflate —
 * lazily, so a route that never opens a project file never pays for it.
 *
 * JSON entries are deflated, image entries stored: bitmaps are already PNG or WEBP, while the
 * document is repetitive JSON that compresses tenfold.
 *
 * ### Why the read guard lives in fflate's filter
 *
 * `unzip` is fully buffered — by the time it returns, every entry is already inflated in memory, so
 * a total measured there is a postmortem rather than a guard. The filter is consulted per entry
 * *before* inflation, using the central directory's declared size, which is the only place a
 * ceiling can actually stop a zip bomb.
 *
 * fflate allocates deflated entries from `originalSize` but stored entries from `size`. Stored
 * entries must declare those identically, so the filter rejects a mismatch before the copy.
 */

/**
 * Ceiling for a single archive, in either direction.
 *
 * Deliberately well under 4 GiB, where ZIP32's 32-bit sizes stop describing the file: fflate reads
 * zip64 records but `zip()` never emits them, so an archive at that boundary would be written
 * structurally invalid rather than rejected.
 */
export const INVK_MAX_ARCHIVE_BYTES = 2 * 1024 * 1024 * 1024;

/** Ceiling on entry count, which bounds the cost of building the entry map. */
export const INVK_MAX_ENTRIES = 20_000;

/** Deflate level for text entries. 6 is fflate's default: the usual ratio/speed knee. */
const TEXT_DEFLATE_LEVEL = 6;

export interface InvkArchiveEntry {
  bytes: Uint8Array;
  /** Text entries are deflated; binary entries are stored. */
  kind: 'text' | 'binary';
}

export type InvkArchiveEntries = ReadonlyMap<string, InvkArchiveEntry>;

export const textEntry = (value: string): InvkArchiveEntry => ({
  bytes: new TextEncoder().encode(value),
  kind: 'text',
});

export const binaryEntry = (bytes: Uint8Array): InvkArchiveEntry => ({ bytes, kind: 'binary' });

export const readEntryText = (bytes: Uint8Array): string => new TextDecoder().decode(bytes);

/** Pack entries into a ZIP blob. Rejects with {@link InvkFormatError} past the size ceiling. */
export const writeArchive = async (entries: InvkArchiveEntries): Promise<Blob> => {
  if (entries.size > INVK_MAX_ENTRIES) {
    throw new InvkFormatError('too-large', `Project archive would hold ${entries.size} entries.`);
  }

  let total = 0;

  for (const entry of entries.values()) {
    total += entry.bytes.byteLength;
  }

  if (total > INVK_MAX_ARCHIVE_BYTES) {
    throw new InvkFormatError('too-large', `Project archive would be ${total} bytes.`);
  }

  const { zip } = await import('fflate');
  const zippable: Record<string, [Uint8Array, { level: 0 | 6 }]> = {};

  for (const [path, entry] of entries) {
    zippable[path] = [entry.bytes, { level: entry.kind === 'text' ? TEXT_DEFLATE_LEVEL : 0 }];
  }

  const packed = await new Promise<Uint8Array>((resolve, reject) => {
    zip(zippable, (error, data) => {
      if (error) {
        reject(error);

        return;
      }

      resolve(data);
    });
  });

  if (packed.byteLength > INVK_MAX_ARCHIVE_BYTES) {
    throw new InvkFormatError('too-large', `Project archive would be ${packed.byteLength} bytes.`);
  }

  return new Blob([packed as BlobPart], { type: INVK_MIME_TYPE });
};

export interface InvkExpansionBudget {
  /** Consulted per entry before it is inflated; `false` keeps it out of memory. */
  accept: (file: { compression: number; name: string; originalSize: number; size: number }) => boolean;
  /** The refusal to raise once `unzip` has settled, or `null` if everything fit. */
  getRefusal: () => InvkFormatError | null;
}

/**
 * The read-side ceiling, consulted during the walk. The refusal is *returned* rather than thrown
 * from `accept`: a throw inside fflate's walk unwinds through its decoder and arrives as a decode
 * failure, so the caller would be told the file was not a project when it was merely too big.
 */
export const createExpansionBudget = (): InvkExpansionBudget => {
  let entryCount = 0;
  let expandedBytes = 0;
  let refusal: InvkFormatError | null = null;

  return {
    accept: (file) => {
      // Directory records carry nothing and only inflate the count toward the
      // guard.
      if (file.name.endsWith('/')) {
        return false;
      }

      entryCount += 1;

      if (entryCount > INVK_MAX_ENTRIES) {
        refusal ??= new InvkFormatError('too-large', `Project archive holds more than ${INVK_MAX_ENTRIES} entries.`);
      } else if (file.compression === 0 && file.size !== file.originalSize) {
        refusal ??= new InvkFormatError('not-a-project', `Stored ZIP entry "${file.name}" has inconsistent sizes.`);
      } else {
        expandedBytes += file.compression === 0 ? file.size : file.originalSize;

        if (expandedBytes > INVK_MAX_ARCHIVE_BYTES) {
          refusal ??= new InvkFormatError('too-large', `Project archive expands past ${INVK_MAX_ARCHIVE_BYTES} bytes.`);
        }
      }

      return refusal === null;
    },
    getRefusal: () => refusal,
  };
};

/**
 * Expand a ZIP into its entries. Anything that is not a readable ZIP surfaces as
 * `not-a-project` rather than an fflate error, because from the caller's side
 * "you picked the wrong file" and "this ZIP has a bad central directory" are the
 * same event.
 */
export const readArchive = async (bytes: Uint8Array): Promise<Map<string, Uint8Array>> => {
  if (bytes.byteLength > INVK_MAX_ARCHIVE_BYTES) {
    throw new InvkFormatError('too-large', `Project archive is ${bytes.byteLength} bytes.`);
  }

  const { unzip } = await import('fflate');
  const budget = createExpansionBudget();
  const expanded = await new Promise<Record<string, Uint8Array>>((resolve, reject) => {
    // `unzip` reports most damage through the callback, but raises some of it — a central directory
    // it cannot walk at all — synchronously. That throw rejects this promise with fflate's own error
    // rather than the one this function promises to raise, so it is caught here too. Otherwise the
    // guarantee holds for a truncated file and quietly fails for a corrupt one.
    try {
      unzip(bytes, { filter: budget.accept }, (error, data) => {
        if (error) {
          reject(new InvkFormatError('not-a-project', error.message));

          return;
        }

        resolve(data);
      });
    } catch (error) {
      reject(new InvkFormatError('not-a-project', error instanceof Error ? error.message : 'Unreadable archive.'));
    }
  });

  const refusal = budget.getRefusal();

  if (refusal) {
    throw refusal;
  }

  return new Map(Object.entries(expanded));
};
