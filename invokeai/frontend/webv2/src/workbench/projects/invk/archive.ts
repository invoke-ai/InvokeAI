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
 *
 * Reading enforces its budget in fflate's entry filter rather than after the
 * fact. `unzip` is fully buffered — by the time it hands back a record of
 * entries, every one of them has already been inflated into memory, so a total
 * measured there is a postmortem, not a guard. The filter is consulted per
 * entry *before* inflation, with the declared uncompressed size from the
 * central directory, which is the only place a ceiling can actually stop a zip
 * bomb rather than describe one.
 *
 * A header that understates its entry's size does not get past this: fflate
 * inflates into a buffer sized from that declaration and fails the decode when
 * the stream overruns it, which surfaces as `not-a-project`.
 */

/**
 * Ceiling for a single archive, in either direction. Large enough for a real
 * project with hundreds of full-resolution layers and a video or two; small
 * enough that a malicious file fails fast rather than exhausting the tab.
 *
 * Deliberately well under 4 GiB. That is where ZIP32's 32-bit sizes and offsets
 * stop being able to describe the file, and fflate writes no zip64 records — it
 * reads them, but `zip()` has no path that emits them — so an archive at that
 * boundary would be written structurally invalid rather than rejected. The
 * browser gives out first anyway: `zip()` holds the entries and the packed
 * output in memory at once.
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
  accept: (file: { name: string; originalSize: number }) => boolean;
  /** The refusal to raise once `unzip` has settled, or `null` if everything fit. */
  getRefusal: () => InvkFormatError | null;
}

/**
 * The read-side ceiling, as a value the ZIP walk consults rather than a total
 * measured afterwards.
 *
 * The refusal is *returned* rather than thrown from `accept`, because a throw
 * inside fflate's own walk unwinds through its decoder and arrives as a decode
 * failure — the caller would be told the file was not a project when what
 * actually happened is that it was too big. Refusing the entry and reporting
 * afterwards keeps the two reasons distinct while still inflating nothing.
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
      expandedBytes += file.originalSize;

      if (entryCount > INVK_MAX_ENTRIES) {
        refusal ??= new InvkFormatError('too-large', `Project archive holds more than ${INVK_MAX_ENTRIES} entries.`);
      } else if (expandedBytes > INVK_MAX_ARCHIVE_BYTES) {
        refusal ??= new InvkFormatError('too-large', `Project archive expands past ${INVK_MAX_ARCHIVE_BYTES} bytes.`);
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
    unzip(bytes, { filter: budget.accept }, (error, data) => {
      if (error) {
        reject(new InvkFormatError('not-a-project', error.message));

        return;
      }

      resolve(data);
    });
  });

  const refusal = budget.getRefusal();

  if (refusal) {
    throw refusal;
  }

  return new Map(Object.entries(expanded));
};
