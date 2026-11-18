/**
 * The mock backend's console transcript, following the read contract of `GET /api/v1/app/server_output`: cursors are
 * `<run>.<revision>.<floor>.<end>`, a read without one answers a bounded snapshot, a cursor from another run answers
 * "restarted", one whose rows were evicted answers "expired", and changes come once each at their latest value, paged
 * with `has_more`. While live, output advances with the clock (one tick per `TICK_MS`, at most `MAX_TICKS_PER_READ`
 * per read): a progress row rewrites in place and log lines with colored levels arrive between bars.
 */

export const SERVER_OUTPUT_TICK_MS = 400;
const MAX_TICKS_PER_READ = 25;
const RETAINED_ROWS = 2_000;
const SNAPSHOT_ROWS = 400;
const PAGE_ROWS = 150;
const CURSOR = /^([0-9a-f]{32})\.(0|[1-9]\d{0,18})\.(0|[1-9]\d{0,18})\.(0|[1-9]\d{0,18})$/;
const PROGRESS_STEPS = 30;
const BAR_WIDTH = 24;

const LEVEL_STYLES = { ERROR: { bold: true, fg: '1' }, INFO: { fg: '6' }, WARNING: { fg: '3' } };

/** A formatter-style line: `[time][InvokeAI]::LEVEL --> message`, the level colored as InvokeAI's console does. */
export const formatLogLine = (time, level, message) => {
  const prefix = `[${time}][InvokeAI]::`;
  const text = `${prefix}${level} --> ${message}`;

  return { styles: [{ length: level.length, offset: prefix.length, ...LEVEL_STYLES[level] }], text };
};

const randomRunId = (seed) => seed.toString(16).padStart(32, '0').slice(-32);

const progressText = (step) => {
  const filled = Math.round((step / PROGRESS_STEPS) * BAR_WIDTH);
  const percent = String(Math.round((step / PROGRESS_STEPS) * 100)).padStart(3);

  return `Denoising ${percent}%|${'█'.repeat(filled)}${' '.repeat(BAR_WIDTH - filled)}| ${step}/${PROGRESS_STEPS}`;
};

export const createServerOutputMock = ({ clock = Date.now, seedLines = [] } = {}) => {
  let runSeed = 1;
  let runId = randomRunId(runSeed);
  let revision = 0;
  let nextId = 1;
  /** Retained rows in ID order, each `{ id, text, styles, truncated, revision }`. */
  let rows = [];
  let lastTick = clock();
  let live = true;
  let step = 0;
  let generation = 0;
  let progressRow = null;

  const write = (text, styles = []) => {
    revision += 1;
    const row = { id: nextId, revision, styles, text, truncated: false };
    nextId += 1;
    rows.push(row);
    if (rows.length > RETAINED_ROWS) {
      rows = rows.slice(rows.length - RETAINED_ROWS);
    }
    return row;
  };

  const rewrite = (row, text) => {
    revision += 1;
    row.revision = revision;
    row.text = text;
  };

  const timeAt = () => new Date(clock()).toISOString().slice(11, 19);

  const tick = () => {
    if (progressRow === null) {
      generation += 1;
      const line = formatLogLine(
        timeAt(),
        'INFO',
        `Executing queue item ${generation}, session mock-session-${generation}`
      );
      write(line.text, line.styles);
      step = 0;
      progressRow = write(progressText(0));
      return;
    }

    step += 1;
    rewrite(progressRow, progressText(step));

    if (step === PROGRESS_STEPS) {
      progressRow = null;
      const line =
        generation % 5 === 0
          ? formatLogLine(timeAt(), 'WARNING', 'Model cache is full; offloading the least recently used model')
          : formatLogLine(timeAt(), 'INFO', `Graph stats: mock-session-${generation} finished in 12.0s`);
      write(line.text, line.styles);
    }
  };

  const advance = () => {
    const now = clock();
    if (!live) {
      lastTick = now;
      return;
    }
    const ticks = Math.min(Math.floor((now - lastTick) / SERVER_OUTPUT_TICK_MS), MAX_TICKS_PER_READ);
    lastTick = ticks === MAX_TICKS_PER_READ ? now : lastTick + ticks * SERVER_OUTPUT_TICK_MS;
    for (let index = 0; index < ticks; index += 1) {
      tick();
    }
  };

  const floor = () => rows[0]?.id ?? nextId;
  const wire = ({ id, styles, text, truncated }) => ({ id, styles, text, truncated });
  const cursorFor = (rev, rowFloor) => `${runId}.${rev}.${rowFloor}.${nextId}`;

  const snapshot = (reset) => {
    const kept = rows.slice(-SNAPSHOT_ROWS);
    const rowFloor = kept[0]?.id ?? nextId;

    return {
      cursor: cursorFor(revision, rowFloor),
      floor_id: rowFloor,
      has_more: false,
      reset,
      rows: kept.map(wire),
      run_id: runId,
      snapshot_truncated: rowFloor > 1,
    };
  };

  for (const line of seedLines) {
    write(line.text, line.styles);
  }

  return {
    /** Appends lines as written by `print()`; a line may carry its own style runs. */
    append: (lines) => {
      for (const line of lines) {
        if (typeof line === 'string') {
          write(line);
        } else {
          write(line.text, line.styles ?? []);
        }
      }
    },
    /** `{ status, body }`: 422 for a cursor this run never issued, otherwise a read per the contract. */
    read: (cursor) => {
      advance();

      if (cursor === null) {
        return { body: snapshot('initial'), status: 200 };
      }

      const match = cursor.length <= 128 ? CURSOR.exec(cursor) : null;
      if (!match) {
        return { body: { detail: 'The cursor is malformed or was not issued by this server' }, status: 422 };
      }

      const [, cursorRun, revisionText, floorText, endText] = match;
      if (cursorRun !== runId) {
        return { body: snapshot('restarted'), status: 200 };
      }

      const cursorRevision = Number(revisionText);
      const cursorFloor = Number(floorText);
      const cursorEnd = Number(endText);
      if (cursorRevision > revision || cursorFloor < 1 || cursorFloor > cursorEnd || cursorEnd > nextId) {
        return { body: { detail: 'The cursor is malformed or was not issued by this server' }, status: 422 };
      }

      const storeFloor = floor();
      if (storeFloor > cursorEnd) {
        return { body: snapshot('expired'), status: 200 };
      }

      const rowFloor = Math.max(cursorFloor, storeFloor);
      const changed = rows
        .filter((row) => row.revision > cursorRevision && row.id >= rowFloor)
        .sort((left, right) => left.revision - right.revision);
      const page = changed.slice(0, PAGE_ROWS);
      const hasMore = changed.length > PAGE_ROWS;
      const pageRevision = hasMore ? page.at(-1).revision : revision;

      return {
        body: {
          cursor: cursorFor(pageRevision, rowFloor),
          floor_id: rowFloor,
          has_more: hasMore,
          reset: null,
          rows: page.sort((left, right) => left.id - right.id).map(wire),
          run_id: runId,
          snapshot_truncated: false,
        },
        status: 200,
      };
    },
    /** A new process run: a new run ID and an empty screen, so held cursors answer "restarted". */
    restart: () => {
      runSeed += 1;
      runId = randomRunId(runSeed);
      revision = 0;
      nextId = 1;
      rows = [];
      progressRow = null;
      lastTick = clock();
    },
    setLive: (value) => {
      advance();
      live = value;
    },
    status: () => ({ live, rows: rows.length, run_id: runId }),
  };
};
