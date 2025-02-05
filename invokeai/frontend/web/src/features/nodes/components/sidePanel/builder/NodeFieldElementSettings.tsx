import { IconButton, Popover, PopoverBody, PopoverContent, PopoverTrigger } from '@invoke-ai/ui-library';
import type { NodeFieldElement } from 'features/nodes/types/workflow';
import { memo } from 'react';
import { PiWrenchFill } from 'react-icons/pi';

export const NodeFieldElementSettings = memo(({ element }: { element: NodeFieldElement }) => {
  return (
    <Popover>
      <PopoverTrigger>
        <IconButton aria-label="settings" icon={<PiWrenchFill />} variant="link" size="sm" alignSelf="stretch" />
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>settings</PopoverBody>
      </PopoverContent>
    </Popover>
  );
});
NodeFieldElementSettings.displayName = 'NodeFieldElementSettings';
