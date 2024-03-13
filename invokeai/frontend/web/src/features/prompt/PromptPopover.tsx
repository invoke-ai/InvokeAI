import { Popover, PopoverAnchor, PopoverBody, PopoverContent } from '@invoke-ai/ui-library';
import { PromptTriggerSelect } from 'features/prompt/PromptTriggerSelect';
import type { PromptPopoverProps } from 'features/prompt/types';
import { memo } from 'react';

export const PromptPopover = memo((props: PromptPopoverProps) => {
  const { onSelect, isOpen, onClose, width, children } = props;

  return (
    <Popover
      isOpen={isOpen}
      onClose={onClose}
      placement="bottom"
      openDelay={0}
      closeDelay={0}
      closeOnBlur={true}
      returnFocusOnClose={false}
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
          <PromptTriggerSelect onClose={onClose} onSelect={onSelect} />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
});

PromptPopover.displayName = 'PromptPopover';
