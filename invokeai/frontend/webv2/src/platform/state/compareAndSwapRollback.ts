/**
 * Roll back only if the slot still equals this mutation's optimistic value; unconditional restore would erase
 * newer writes.
 */

/** What a rollback needs to know about a slot: what was there, and what we put there. */
export interface CompareAndSwapEntry<Value> {
  /** The value this mutation painted. The slot must still hold it to be revertible. */
  after: Value;
  /** The value to restore. */
  before: Value;
}

export interface CompareAndSwapOptions {
  /** Enable only when undefined means no local copy, never when another writer could deliberately clear the slot. */
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
