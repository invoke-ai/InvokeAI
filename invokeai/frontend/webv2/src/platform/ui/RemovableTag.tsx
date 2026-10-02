import type { ReactNode } from 'react';

import { Tag } from '@chakra-ui/react';
import { useCallback, useRef, useState } from 'react';

import { Tooltip } from './Tooltip';

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
  const [isLabelTipOpen, setIsLabelTipOpen] = useState(false);
  // Only a truncated label needs its full text on hover.
  const handleLabelTipChange = useCallback(({ open }: { open: boolean }) => {
    const label = labelRef.current;
    setIsLabelTipOpen(open && label !== null && label.scrollWidth > label.clientWidth);
  }, []);

  return (
    <Tag.Root flexShrink={0} maxW="full" minW="0" size="sm" variant="surface">
      <Tooltip content={children} open={isLabelTipOpen} onOpenChange={handleLabelTipChange}>
        <Tag.Label ref={labelRef} truncate>
          {children}
        </Tag.Label>
      </Tooltip>
      <Tag.EndElement>
        <Tooltip content={removeLabel}>
          <Tag.CloseTrigger aria-label={removeLabel} disabled={disabled} onClick={onRemove} />
        </Tooltip>
      </Tag.EndElement>
    </Tag.Root>
  );
};
