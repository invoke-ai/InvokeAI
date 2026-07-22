/**
 * Frecency record of executed palette entries, persisted per browser (not per
 * project — muscle memory should not reset when switching projects). Reads
 * return ids ranked by use count decayed by recency; stored ids can go stale
 * (commands renamed, widgets closed), so readers filter them against the live
 * entry list at render time.
 */

const STORAGE_KEY = 'invokeai:v7:webv2:palette-recents';
const MAX_RECORDS = 40;

const HOUR_MS = 60 * 60 * 1000;
const DAY_MS = 24 * HOUR_MS;
const WEEK_MS = 7 * DAY_MS;

interface RecentUse {
  id: string;
  lastUsedAt: number;
  uses: number;
}

const isBrowser = (): boolean => typeof window !== 'undefined' && typeof window.localStorage !== 'undefined';

const parseRecords = (raw: string | null): RecentUse[] => {
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as unknown;

    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.flatMap((item): RecentUse[] => {
      // Earlier builds stored a plain id ring buffer.
      if (typeof item === 'string') {
        return [{ id: item, lastUsedAt: 0, uses: 1 }];
      }

      if (
        item !== null &&
        typeof item === 'object' &&
        typeof (item as RecentUse).id === 'string' &&
        typeof (item as RecentUse).lastUsedAt === 'number' &&
        typeof (item as RecentUse).uses === 'number'
      ) {
        return [item as RecentUse];
      }

      return [];
    });
  } catch {
    return [];
  }
};

const readRecords = (): RecentUse[] => (isBrowser() ? parseRecords(window.localStorage.getItem(STORAGE_KEY)) : []);

/** Recency multiplier: recent use outweighs raw counts, but never zeroes them. */
const recencyWeight = (ageMs: number): number => {
  if (ageMs < HOUR_MS) {
    return 4;
  }
  if (ageMs < DAY_MS) {
    return 2;
  }
  if (ageMs < WEEK_MS) {
    return 1;
  }

  return 0.5;
};

const frecencyScore = (record: RecentUse, now: number): number =>
  record.uses * recencyWeight(Math.max(0, now - record.lastUsedAt));

/** Entry ids ranked by frecency, best first. */
export const getRecentEntryIds = (): string[] => {
  const now = Date.now();

  return readRecords()
    .map((record) => ({ record, score: frecencyScore(record, now) }))
    .sort((left, right) => right.score - left.score || right.record.lastUsedAt - left.record.lastUsedAt)
    .map(({ record }) => record.id);
};

export const recordRecentEntry = (entry: Pick<PaletteEntry, 'id' | 'isPersistentRecent'>): void => {
  if (!entry.isPersistentRecent || !isBrowser()) {
    return;
  }

  const records = readRecords();
  const existing = records.find((record) => record.id === entry.id);
  const next: RecentUse[] = [
    { id: entry.id, lastUsedAt: Date.now(), uses: (existing?.uses ?? 0) + 1 },
    ...records.filter((record) => record.id !== entry.id),
  ].slice(0, MAX_RECORDS);

  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Quota or private-mode failures are non-fatal; recents are a convenience.
  }
};
import type { PaletteEntry } from './entries';
