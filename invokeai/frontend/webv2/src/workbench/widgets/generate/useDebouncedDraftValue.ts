import { useEffect, useLayoutEffect, useRef, useState } from 'react';

interface UseDebouncedDraftValueOptions<Value> {
  value: Value;
  delayMs: number;
  onCommit: (value: Value) => void;
  isEqual?: (left: Value, right: Value) => boolean;
  flushOnUnmount?: boolean;
  resetKey?: string;
}

interface DebouncedDraftValue<Value> {
  draftValue: Value;
  setDraftValue: (value: Value) => void;
  commitDraftValue: (value: Value) => void;
  replaceDraftValue: (value: Value) => void;
  flushDraftValue: () => void;
  hasPendingDraft: () => boolean;
}

export const useDebouncedDraftValue = <Value>({
  delayMs,
  flushOnUnmount = true,
  isEqual = Object.is,
  onCommit,
  resetKey,
  value,
}: UseDebouncedDraftValueOptions<Value>): DebouncedDraftValue<Value> => {
  const [draftValue, setDraftValueState] = useState(value);
  const draftValueRef = useRef(value);
  const sourceValueRef = useRef(value);
  const resetKeyRef = useRef(resetKey);
  const hasPendingDraftRef = useRef(false);
  const pendingDraftRef = useRef<Value | null>(null);
  const timeoutRef = useRef<number | null>(null);
  const flushOnUnmountRef = useRef(flushOnUnmount);
  const onCommitRef = useRef(onCommit);

  flushOnUnmountRef.current = flushOnUnmount;
  onCommitRef.current = onCommit;

  const setDraft = (nextValue: Value) => {
    draftValueRef.current = nextValue;
    setDraftValueState(nextValue);
  };

  const clearPendingCommit = () => {
    if (timeoutRef.current !== null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
  };

  const clearPendingDraft = () => {
    clearPendingCommit();
    hasPendingDraftRef.current = false;
    pendingDraftRef.current = null;
  };

  const flushDraftValue = () => {
    if (!hasPendingDraftRef.current) {
      return;
    }

    const valueToCommit = pendingDraftRef.current as Value;

    clearPendingDraft();
    onCommitRef.current(valueToCommit);
  };

  useLayoutEffect(() => {
    if (resetKeyRef.current !== resetKey) {
      resetKeyRef.current = resetKey;
      clearPendingDraft();
      sourceValueRef.current = value;
      setDraft(value);
      return;
    }

    if (isEqual(sourceValueRef.current, value)) {
      return;
    }

    sourceValueRef.current = value;

    if (!hasPendingDraftRef.current) {
      setDraft(value);
      return;
    }

    if (isEqual(pendingDraftRef.current as Value, value)) {
      clearPendingDraft();
      setDraft(value);
      return;
    }

    clearPendingDraft();
    setDraft(value);
  }, [isEqual, resetKey, value]);

  useEffect(
    () => () => {
      clearPendingCommit();

      if (flushOnUnmountRef.current) {
        flushDraftValue();
        return;
      }

      clearPendingDraft();
    },
    []
  );

  const setDraftValue = (nextValue: Value) => {
    setDraft(nextValue);
    hasPendingDraftRef.current = true;
    pendingDraftRef.current = nextValue;
    clearPendingCommit();

    timeoutRef.current = window.setTimeout(flushDraftValue, delayMs);
  };

  const commitDraftValue = (nextValue: Value) => {
    clearPendingDraft();
    setDraft(nextValue);
    onCommitRef.current(nextValue);
  };

  const replaceDraftValue = (nextValue: Value) => {
    clearPendingDraft();
    setDraft(nextValue);
  };

  return {
    commitDraftValue,
    draftValue,
    flushDraftValue,
    hasPendingDraft: () => hasPendingDraftRef.current,
    replaceDraftValue,
    setDraftValue,
  };
};
