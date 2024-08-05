import {
  Button,
  Popover,
  PopoverBody,
  PopoverContent,
  PopoverTrigger,
} from '@invoke-ai/ui-library';

import { StylePresetMenu } from './StylePresetMenu';

export const StylePresetMenuTrigger = () => {
  return (
    <Popover isLazy>
      <PopoverTrigger>
        <Button size="sm">Style Presets</Button>
      </PopoverTrigger>
      <PopoverContent>
        <PopoverBody>
          <StylePresetMenu />
        </PopoverBody>
      </PopoverContent>
    </Popover>
  );
};
