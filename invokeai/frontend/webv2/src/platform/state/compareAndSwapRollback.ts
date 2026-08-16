/**
 * The one rule for undoing an optimistic write: restore a slot only if it still
 * holds exactly what this mutation painted there.
 *
 * Between the optimistic write and the server's rejection, anything may have
 * written to the same slot — a generation completing, a refetch landing, a
 * second mutation on the same item. Restoring unconditionally throws that newer
 * work away, and it fails silently.
 *
 * This lived in four places (twice in the gallery query cache, plus the widget-
 * value snapshot and the per-item rollback), each with a comment pointing at the
 * others. One wrong copy is data loss, so there is one copy.
 */

/** What a rollback needs to know about a slot: what was there, and what we put there. */
export interface CompareAndSwapEntry<Value> {
  /** The value this mutation painted. The slot must still hold it to be revertible. */
  after: Value;
  /** The value to restore. */
  before: Value;
}

export interface CompareAndSwapOptions {
  /**
   * Treat a slot that reads `undefined` as revertible.
   *
   * Correct where `undefined` means "no local copy of this exists" rather than
   * "someone cleared it" — a per-item lookup that simply misses, for instance.
   * Wrong where `undefined` is a value another writer could have set.
   */
  treatUnknownAsUnclaimed?: boolean;
}

/** Whether `current` still reads as the value this mutation painted. */
export const isSlotUnclaimed = <Value>(
  current: Value | undefined,
  painted: Value,
  { treatUnknownAsUnclaimed = false }: CompareAndSwapOptions = {}
): boolean => current === painted || (treatUnknownAsUnclaimed && current === undefined);

/** The subset of `entries` whose slots no newer writer has claimed. */
export const selectUnclaimedEntries = <Value, Entry extends CompareAndSwapEntry<Value>>(
  entries: readonly Entry[],
  readCurrent: (entry: Entry) => Value | undefined,
  options?: CompareAndSwapOptions
): Entry[] => entries.filter((entry) => isSlotUnclaimed(readCurrent(entry), entry.after, options));

/** Restore every entry whose slot is still ours, and skip the rest. */
export const rollBackUnclaimedEntries = <Value, Entry extends CompareAndSwapEntry<Value>>(
  entries: readonly Entry[],
  readCurrent: (entry: Entry) => Value | undefined,
  restore: (entry: Entry) => void,
  options?: CompareAndSwapOptions
): void => {
  for (const entry of selectUnclaimedEntries(entries, readCurrent, options)) {
    restore(entry);
  }
};
