/**
 * The one rule for undoing an optimistic write: put a slot back only if it
 * still holds exactly what this mutation painted there.
 *
 * An optimistic mutation writes a value, the server rejects it, and the
 * rollback wants the old value back. Between those two moments anything may
 * have written to the same slot — a generation completing, a refetch landing,
 * the person re-selecting, a second mutation on the same item. Restoring
 * unconditionally would throw that newer work away, and the failure is silent:
 * the UI shows stale state that no longer corresponds to anything.
 *
 * So the restore is a compare-and-swap. `after` is what we painted; if the slot
 * still reads as `after`, we are the last writer and may safely revert. If it
 * reads as anything else, someone newer owns it and we leave it alone.
 *
 * This lived in three separate places — the gallery query cache, the per-project
 * widget-value snapshot, and the per-item board/starred rollback — each with its
 * own copy of the check and a comment pointing at the other two. One wrong copy
 * is a data-loss bug, so there is one copy.
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
