import { INVK_MIME_TYPE, InvkFormatError } from './format';

/**
 * The ZIP seam. The only module in the app that knows `.invk` is a ZIP, and the
 * only one that imports fflate.
 *
 * fflate is loaded with `await import()` at call time, exactly as `psdExport.ts`
 * loads ag-psd and `wildcardFiles.ts` loads `yaml`. Reading or writing a project
 * file is a deliberate, occasional act; the bytes that make it possible have no
 * business in the graph a person downloads to look at their projects. The
 * architecture budget enforces this — a new package appearing in a route's
 * initial chunk fails the gate on the source-owner set alone, before anyone
 * looks at kilobytes.
 *
 * ### Compression
 *
 * JSON entries are deflated; image entries are stored. Layer bitmaps and
 * generated results are already PNG or WEBP, so deflating them spends CPU
 * proportional to the archive to save a fraction of a percent. The project
 * document, by contrast, is repetitive JSON that routinely compresses tenfold.
 *
 * ### Guards
 *
 * Both directions are bounded. An `.invk` is a file someone was handed, and an
 * unbounded `unzip` on a hostile one is a way to be handed an out-of-memory
 * crash instead of an error message.
 */

/**
 * Ceiling for a single archive, in either direction. Large enough for a real
 * project with hundreds of full-resolution layers; small enough that a
 * malicious file fails fast rather than exhausting the tab.
 */
export const INVK_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024;

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

  return new Blob([packed as BlobPart], { type: INVK_MIME_TYPE });
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
  const expanded = await new Promise<Record<string, Uint8Array>>((resolve, reject) => {
    unzip(bytes, (error, data) => {
      if (error) {
        reject(new InvkFormatError('not-a-project', error.message));

        return;
      }

      resolve(data);
    });
  });

  const entries = new Map<string, Uint8Array>();
  let total = 0;

  for (const [path, entry] of Object.entries(expanded)) {
    // Directory records come through as zero-length entries; they carry nothing
    // and only inflate the count toward the guard.
    if (path.endsWith('/')) {
      continue;
    }

    if (entries.size >= INVK_MAX_ENTRIES) {
      throw new InvkFormatError('too-large', `Project archive holds more than ${INVK_MAX_ENTRIES} entries.`);
    }

    total += entry.byteLength;

    if (total > INVK_MAX_ARCHIVE_BYTES) {
      throw new InvkFormatError('too-large', `Project archive expands past ${INVK_MAX_ARCHIVE_BYTES} bytes.`);
    }

    entries.set(path, entry);
  }

  return entries;
};
