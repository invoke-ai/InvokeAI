import { Popover, PopoverAnchor, PopoverBody, PopoverContent } from '@invoke-ai/ui-library';
import { EmbeddingSelect } from 'features/embedding/EmbeddingSelect';
import type { EmbeddingPopoverProps } from 'features/embedding/types';
import { memo } from 'react';

export const EmbeddingPopover = memo((props: EmbeddingPopoverProps) => {
  const { onSelect, isOpen, onClose, width, children } = props;

  return (
    <Popover
      isOpen={isOpen}
      onClose={onClose}
      placement="bottom"
      openDelay={0}
      closeDelay={0}
      closeOnBlur={true}
      returnFocusOnClose={true}
      isLazy
    >
      <PopoverAnchor>{children}</PopoverAnchor>
      <PopoverContent
        p={0}
        insetBlockStart={-1}
        shadow="dark-lg"
        borderColor="invokeBlue.300"
        borderWidth="2px"
        borderStyle="solid"
      >
        <PopoverBody p={0} width={`calc(${width}px - 0.25rem)`}>
          <EmbeddingSelect onClose={onClose} onSelect={onSelect} />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

EmbeddingPopover.displayName = 'EmbeddingPopover';
