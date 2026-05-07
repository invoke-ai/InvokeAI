export type PromptWorkbenchKeyboardTarget = 'autocomplete' | 'fixed_values';
export type PromptWorkbenchKeyboardIntent =
  | 'dismiss'
  | 'insert_fixed_value'
  | 'insert_wildcard'
  | 'next'
  | 'open_fixed_values'
  | 'previous';

export const getNextNavigationIndex = ({
  currentIndex,
  direction,
  itemCount,
}: {
  currentIndex: number;
  direction: 'next' | 'previous';
  itemCount: number;
}): number => {
  if (itemCount <= 0) {
    return 0;
  }

  const delta = direction === 'next' ? 1 : -1;
  return (currentIndex + delta + itemCount) % itemCount;
};

export const clampNavigationIndex = (index: number, itemCount: number): number => {
  if (itemCount <= 0) {
    return 0;
  }
  return Math.min(Math.max(index, 0), itemCount - 1);
};

export const getPromptWorkbenchKeyboardIntent = ({
  key,
  shiftKey,
  target,
}: {
  key: string;
  shiftKey: boolean;
  target: PromptWorkbenchKeyboardTarget;
}): PromptWorkbenchKeyboardIntent | null => {
  if (key === 'Escape') {
    return 'dismiss';
  }

  if (key === 'ArrowDown') {
    return 'next';
  }

  if (key === 'ArrowUp') {
    return 'previous';
  }

  if (target === 'fixed_values' && key === 'Enter') {
    return 'insert_fixed_value';
  }

  if (target === 'autocomplete' && key === 'Enter' && shiftKey) {
    return 'open_fixed_values';
  }

  if (target === 'autocomplete' && (key === 'Enter' || key === 'Tab')) {
    return 'insert_wildcard';
  }

  return null;
};
