import type { ReactNode } from 'react';

import { Tag } from '@chakra-ui/react';
import { useCallback, useRef } from 'react';

import { ClippedTooltip, Tooltip } from './Tooltip';

export interface RemovableTagProps {
  children: ReactNode;
  /** Names the remove control and its tooltip, e.g. "Show all projects". */
  removeLabel: string;
  onRemove: () => void;
  disabled?: boolean;
}

/** A compact chip for an applied filter or value, with an icon-only remove control. */
export const RemovableTag = ({ children, disabled, onRemove, removeLabel }: RemovableTagProps) => {
  const labelRef = useRef<HTMLSpanElement | null>(null);
  // Only a truncated label needs its full text on hover.
  const isLabelTruncated = useCallback(() => {
    const label = labelRef.current;
    return label !== null && label.scrollWidth > label.clientWidth;
  }, []);

  return (
    <Tag.Root flexShrink={0} maxW="full" minW="0" variant="surface">
      <ClippedTooltip content={children} isClipped={isLabelTruncated}>
        <Tag.Label ref={labelRef} truncate>
          {children}
        </Tag.Label>
      </ClippedTooltip>
      <Tag.EndElement>
        <Tooltip content={removeLabel}>
          <Tag.CloseTrigger aria-label={removeLabel} disabled={disabled} onClick={onRemove} />
        </Tooltip>
      </Tag.EndElement>
    </Tag.Root>
  );
};
