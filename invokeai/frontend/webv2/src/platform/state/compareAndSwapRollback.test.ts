import { describe, expect, it } from 'vitest';

import { isSlotUnclaimed, rollBackUnclaimedEntries, selectUnclaimedEntries } from './compareAndSwapRollback';

describe('isSlotUnclaimed', () => {
  it('is true only while the slot still holds what we painted', () => {
    const painted = { id: 'painted' };

    expect(isSlotUnclaimed(painted, painted)).toBe(true);
    expect(isSlotUnclaimed({ id: 'painted' }, painted)).toBe(false);
    expect(isSlotUnclaimed({ id: 'newer' }, painted)).toBe(false);
  });

  it('rejects an unknown slot unless the caller says unknown means absent', () => {
    // By default `undefined` may be a value a newer writer set, so reverting
    // over it would be the data loss the rule exists to prevent. Opting in
    // still does not excuse a slot someone else actually claimed.
    expect(isSlotUnclaimed(undefined, 'painted')).toBe(false);
    expect(isSlotUnclaimed(undefined, 'painted', { treatUnknownAsUnclaimed: true })).toBe(true);
    expect(isSlotUnclaimed('newer', 'painted', { treatUnknownAsUnclaimed: true })).toBe(false);
  });
});

describe('selectUnclaimedEntries', () => {
  const entries = [
    { after: 'painted-a', before: 'original-a', slot: 'a' },
    { after: 'painted-b', before: 'original-b', slot: 'b' },
  ];

  it('drops entries a newer writer has taken over', () => {
    const current: Record<string, string> = { a: 'painted-a', b: 'someone-else' };

    expect(selectUnclaimedEntries(entries, (entry) => current[entry.slot])).toEqual([entries[0]]);
  });

  it('keeps every entry when nothing else wrote', () => {
    const current: Record<string, string> = { a: 'painted-a', b: 'painted-b' };

    expect(selectUnclaimedEntries(entries, (entry) => current[entry.slot])).toEqual(entries);
  });
});

describe('rollBackUnclaimedEntries', () => {
  it('restores only the slots still holding our own write', () => {
    const slots: Record<string, string> = { a: 'painted-a', b: 'someone-else' };

    rollBackUnclaimedEntries(
      [
        { after: 'painted-a', before: 'original-a', slot: 'a' },
        { after: 'painted-b', before: 'original-b', slot: 'b' },
      ],
      (entry) => slots[entry.slot],
      (entry) => {
        slots[entry.slot] = entry.before;
      }
    );

    expect(slots).toEqual({ a: 'original-a', b: 'someone-else' });
  });

  it('does nothing when every slot has moved on', () => {
    const slots: Record<string, string> = { a: 'newer' };

    rollBackUnclaimedEntries(
      [{ after: 'painted-a', before: 'original-a', slot: 'a' }],
      (entry) => slots[entry.slot],
      (entry) => {
        slots[entry.slot] = entry.before;
      }
    );

    expect(slots).toEqual({ a: 'newer' });
  });
});
