import type { components } from '@api/schema';

export type ServerOutputRead = components['schemas']['ServerOutputRead'];
export type ServerOutputRow = components['schemas']['ServerOutputRow'];
export type ServerOutputStyleRun = components['schemas']['ServerOutputStyleRun'];

/** Rows share a block by ID range, so a block keeps its key while rows around it come and go. */
export const TRANSCRIPT_BLOCK_ROWS = 64;

export interface TranscriptLimits {
  readonly maxRows: number;
  /** Approximate wire size, the unit of the server's own retention budget. */
  readonly maxSize: number;
}

/**
 * The server retains about 2 MiB of rows and a snapshot carries at most 256 KiB, so the size bound mirrors the
 * server's. The row bound keeps the DOM of many short rows manageable.
 */
export const TRANSCRIPT_LIMITS: TranscriptLimits = { maxRows: 20_000, maxSize: 2 * 1024 * 1024 };

export interface TranscriptBlock {
  readonly key: number;
  /** Ascending by ID. */
  readonly rows: readonly ServerOutputRow[];
  readonly size: number;
}

export type TranscriptReset = 'expired' | 'restarted';

export interface Transcript {
  readonly runId: string | null;
  /** Ascending by key; unchanged blocks keep their identity across reads. */
  readonly blocks: readonly TranscriptBlock[];
  readonly rowCount: number;
  readonly size: number;
  /** Earlier output of this run is not held: a truncated snapshot, the server's retention or this bound left it out. */
  readonly truncated: boolean;
  /**
   * Why the rows last replaced what this view held: the server restarted, or the history it needed expired. `serial`
   * tells one replacement from the next, so a view can dismiss a notice without hiding a later one.
   */
  readonly reset: { readonly reason: TranscriptReset; readonly serial: number } | null;
}

export const EMPTY_TRANSCRIPT: Transcript = Object.freeze({
  blocks: [],
  reset: null,
  rowCount: 0,
  runId: null,
  size: 0,
  truncated: false,
});

const ROW_OVERHEAD = 48;
const STYLE_RUN_SIZE = 64;

const rowSize = (row: ServerOutputRow): number => ROW_OVERHEAD + row.text.length + row.styles.length * STYLE_RUN_SIZE;

const blockKey = (id: number): number => Math.floor(id / TRANSCRIPT_BLOCK_ROWS);

const createBlock = (key: number, rows: readonly ServerOutputRow[]): TranscriptBlock => ({
  key,
  rows,
  size: rows.reduce((total, row) => total + rowSize(row), 0),
});

const sortRows = (rows: readonly ServerOutputRow[]): readonly ServerOutputRow[] => {
  for (let index = 1; index < rows.length; index += 1) {
    if (rows[index]!.id <= rows[index - 1]!.id) {
      // Keep the last value of a repeated ID, as an upsert would.
      const byId = new Map(rows.map((row) => [row.id, row]));
      return [...byId.values()].sort((left, right) => left.id - right.id);
    }
  }

  return rows;
};

const groupIntoBlocks = (rows: readonly ServerOutputRow[]): TranscriptBlock[] => {
  const blocks: TranscriptBlock[] = [];
  let start = 0;

  for (let index = 1; index <= rows.length; index += 1) {
    if (index === rows.length || blockKey(rows[index]!.id) !== blockKey(rows[start]!.id)) {
      blocks.push(createBlock(blockKey(rows[start]!.id), rows.slice(start, index)));
      start = index;
    }
  }

  return blocks;
};

/** Merges two ascending row lists; `incoming` replaces rows with the same ID. */
const mergeRows = (
  existing: readonly ServerOutputRow[],
  incoming: readonly ServerOutputRow[]
): readonly ServerOutputRow[] => {
  const merged: ServerOutputRow[] = [];
  let left = 0;
  let right = 0;

  while (left < existing.length || right < incoming.length) {
    const held = existing[left];
    const next = incoming[right];

    if (next === undefined || (held !== undefined && held.id < next.id)) {
      merged.push(held!);
      left += 1;
    } else {
      merged.push(next);
      right += 1;
      if (held?.id === next.id) {
        left += 1;
      }
    }
  }

  return merged;
};

const upsert = (blocks: readonly TranscriptBlock[], rows: readonly ServerOutputRow[]): readonly TranscriptBlock[] => {
  if (rows.length === 0) {
    return blocks;
  }

  const incoming = groupIntoBlocks(rows);
  const merged: TranscriptBlock[] = [];
  let index = 0;

  for (const block of incoming) {
    while (index < blocks.length && blocks[index]!.key < block.key) {
      merged.push(blocks[index]!);
      index += 1;
    }

    const held = blocks[index];

    if (held?.key === block.key) {
      merged.push(createBlock(block.key, mergeRows(held.rows, block.rows)));
      index += 1;
    } else {
      merged.push(block);
    }
  }

  merged.push(...blocks.slice(index));

  return merged;
};

/** Drops rows below `floorId`; returns the same array when nothing is below it. */
const dropBelow = (blocks: readonly TranscriptBlock[], floorId: number): readonly TranscriptBlock[] => {
  let index = 0;

  while (index < blocks.length && blocks[index]!.rows.at(-1)!.id < floorId) {
    index += 1;
  }

  const first = blocks[index];

  if (first === undefined) {
    return index === 0 ? blocks : [];
  }

  const keep = first.rows.findIndex((row) => row.id >= floorId);

  if (keep === 0) {
    return index === 0 ? blocks : blocks.slice(index);
  }

  return [createBlock(first.key, first.rows.slice(keep)), ...blocks.slice(index + 1)];
};

const sumRows = (blocks: readonly TranscriptBlock[]): number =>
  blocks.reduce((total, block) => total + block.rows.length, 0);

const sumSize = (blocks: readonly TranscriptBlock[]): number => blocks.reduce((total, block) => total + block.size, 0);

/** Drops the oldest rows until both bounds hold. */
const enforceLimits = (blocks: readonly TranscriptBlock[], limits: TranscriptLimits): readonly TranscriptBlock[] => {
  let rowCount = sumRows(blocks);
  let size = sumSize(blocks);

  if (rowCount <= limits.maxRows && size <= limits.maxSize) {
    return blocks;
  }

  const kept = [...blocks];

  while (kept.length > 0 && (rowCount > limits.maxRows || size > limits.maxSize)) {
    const first = kept[0]!;
    const rowExcess = rowCount - limits.maxRows;

    if (first.rows.length <= rowExcess || size - first.size >= limits.maxSize) {
      kept.shift();
      rowCount -= first.rows.length;
      size -= first.size;
      continue;
    }

    // Trim the first block row by row until both bounds hold.
    let drop = Math.max(0, rowExcess);
    let dropped = first.rows.slice(0, drop).reduce((total, row) => total + rowSize(row), 0);

    while (size - dropped > limits.maxSize && drop < first.rows.length) {
      dropped += rowSize(first.rows[drop]!);
      drop += 1;
    }

    kept[0] = createBlock(first.key, first.rows.slice(drop));
    rowCount -= drop;
    size -= dropped;
  }

  return kept;
};

const createTranscript = (
  runId: string,
  blocks: readonly TranscriptBlock[],
  truncated: boolean,
  reset: Transcript['reset']
): Transcript => ({
  blocks,
  reset,
  rowCount: sumRows(blocks),
  runId,
  size: sumSize(blocks),
  truncated,
});

/**
 * Applies one read. A reset replaces everything with its rows; otherwise rows are upserted by ID and rows below the
 * floor are dropped. Returns `transcript` itself when the read changed nothing.
 */
export const applyServerOutputRead = (
  transcript: Transcript,
  read: ServerOutputRead,
  limits: TranscriptLimits = TRANSCRIPT_LIMITS
): Transcript => {
  if (read.reset !== null) {
    const blocks = groupIntoBlocks(sortRows(read.rows));
    const bounded = enforceLimits(blocks, limits);

    return createTranscript(
      read.run_id,
      bounded,
      read.snapshot_truncated || bounded !== blocks,
      read.reset === 'initial' ? null : { reason: read.reset, serial: (transcript.reset?.serial ?? 0) + 1 }
    );
  }

  const upserted = upsert(transcript.blocks, sortRows(read.rows));
  const floored = dropBelow(upserted, read.floor_id);
  const bounded = enforceLimits(floored, limits);

  if (bounded === transcript.blocks && read.run_id === transcript.runId) {
    return transcript;
  }

  return createTranscript(
    read.run_id,
    bounded,
    transcript.truncated || floored !== upserted || bounded !== floored,
    transcript.reset
  );
};

export function* iterateRows(transcript: Transcript): Generator<ServerOutputRow> {
  for (const block of transcript.blocks) {
    yield* block.rows;
  }
}

/** Plain text of rows `fromId` through `toId`, cutting the first and last at UTF-16 offsets. */
export const getTranscriptText = (
  transcript: Transcript,
  range?: { fromId: number; fromOffset: number; toId: number; toOffset: number }
): string => {
  const lines: string[] = [];

  for (const row of iterateRows(transcript)) {
    if (range && (row.id < range.fromId || row.id > range.toId)) {
      continue;
    }

    const start = range && row.id === range.fromId ? range.fromOffset : 0;
    const end = range && row.id === range.toId ? range.toOffset : row.text.length;

    lines.push(row.text.slice(start, end));
  }

  return lines.join('\n');
};
