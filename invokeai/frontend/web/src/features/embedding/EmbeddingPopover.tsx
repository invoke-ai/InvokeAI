import {
  InvPopover,
  InvPopoverAnchor,
  InvPopoverBody,
  InvPopoverContent,
} from 'common/components/InvPopover/wrapper';
import { EmbeddingSelect } from 'features/embedding/EmbeddingSelect';
import type { EmbeddingPopoverProps } from 'features/embedding/types';
import { memo } from 'react';
import { PARAMETERS_PANEL_WIDTH } from 'theme/util/constants';

export const EmbeddingPopover = memo((props: EmbeddingPopoverProps) => {
  const {
    onSelect,
    isOpen,
    onClose,
    width = PARAMETERS_PANEL_WIDTH,
    children,
  } = props;

  return (
    <InvPopover
      isOpen={isOpen}
      onClose={onClose}
      placement="bottom"
      openDelay={0}
      closeDelay={0}
      closeOnBlur={true}
      returnFocusOnClose={true}
      isLazy
    >
      <InvPopoverAnchor>{children}</InvPopoverAnchor>
      <InvPopoverContent
        p={0}
        insetBlockStart={-1}
        shadow="dark-lg"
        borderColor="invokeBlue.300"
        borderWidth="2px"
        borderStyle="solid"
      >
        <InvPopoverBody p={0} width={`calc(${width}px - 0.25rem)`}>
          <EmbeddingSelect onClose={onClose} onSelect={onSelect} />
        </InvPopoverBody>
      </InvPopoverContent>
    </InvPopover>
  );
});

EmbeddingPopover.displayName = 'EmbeddingPopover';
