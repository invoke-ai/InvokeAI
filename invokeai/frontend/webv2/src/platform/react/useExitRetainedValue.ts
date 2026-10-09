import { useCallback, useState } from 'react';

interface Retained<Value> {
  generation: number;
  isOpen: boolean;
  value: Value | null;
}

/**
 * Keep an overlay's subject while its exit animation plays. Callers that render `{subject ? <Dialog open /> : null}`
 * unmount the dialog the moment it closes, so it vanishes instead of animating out. Render while `value` is non-null,
 * pass `isOpen` as the dialog's open state, `release` as its `onExitComplete`, and `generation` as its key so each
 * open starts a fresh form.
 *
 * `generation` changes only when the overlay opens, so a subject that changes while open (a refreshed record)
 * updates `value` without remounting. An object subject must keep its identity while its inputs are unchanged
 * (memoize it): a new object on every render never settles. Release relies on the overlay reporting its exit:
 * one that closes before it ever rendered open (a lazy chunk still loading) reports nothing, so the last value stays
 * mounted, closed and inert, until the next open replaces it.
 */
export const useExitRetainedValue = <Value extends NonNullable<unknown>>(subject: Value | null) => {
  const [retained, setRetained] = useState<Retained<Value>>(() => ({
    generation: subject === null ? 0 : 1,
    isOpen: subject !== null,
    value: subject,
  }));
  const isOpen = subject !== null;

  // Adjusted during render (React's pattern for state derived from a changing prop), so the overlay never renders a
  // frame with a stale open state.
  if (isOpen && !retained.isOpen) {
    setRetained({ generation: retained.generation + 1, isOpen: true, value: subject });
  } else if (isOpen && retained.value !== subject) {
    // Track the latest open subject, so the closing overlay shows what it last showed.
    setRetained({ ...retained, value: subject });
  } else if (!isOpen && retained.isOpen) {
    setRetained({ ...retained, isOpen: false });
  }

  const release = useCallback(
    () => setRetained((current) => (current.isOpen ? current : { ...current, value: null })),
    []
  );

  return { generation: retained.generation, isOpen, release, value: isOpen ? subject : retained.value };
};

/** `useExitRetainedValue` for an overlay with no subject: `isMounted` stays true until its exit animation ends. */
export const useExitPresence = (isOpen: boolean) => {
  const { generation, release, value } = useExitRetainedValue(isOpen ? true : null);

  return { generation, isMounted: value !== null, isOpen, release };
};
