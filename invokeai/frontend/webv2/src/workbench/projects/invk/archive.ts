import { INVK_MIME_TYPE, InvkFormatError } from './format';

/**
 * Lazy ZIP boundary: deflate JSON and store compressed media. Enforce declared-size budgets in the filter before
 * buffered inflation. Stored entries require size === originalSize because fflate allocates them from size.
 */

/** Stay below ZIP32 overflow; the writer does not emit zip64. */
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
 * Return size refusals instead of throwing inside the filter, which would misclassify oversized archives as
 * corrupt.
 */
export const createExpansionBudget = (): InvkExpansionBudget => {
  let entryCount = 0;
  let expandedBytes = 0;
  let refusal: InvkFormatError | null = null;

  return {
    accept: (file) => {
      // Empty directory records do not consume the entry budget.
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

/** Corrupt and non-ZIP inputs share the same typed refusal. */
export const readArchive = async (bytes: Uint8Array): Promise<Map<string, Uint8Array>> => {
  if (bytes.byteLength > INVK_MAX_ARCHIVE_BYTES) {
    throw new InvkFormatError('too-large', `Project archive is ${bytes.byteLength} bytes.`);
  }

  const { unzip } = await import('fflate');
  const budget = createExpansionBudget();
  const expanded = await new Promise<Record<string, Uint8Array>>((resolve, reject) => {
    // Normalize both synchronous throws and asynchronous unzip errors.
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
