import { IconButton, MenuButton } from '@invoke-ai/ui-library';
import { stopPropagation } from 'common/util/stopPropagation';
import { memo } from 'react';
import { PiDotsThreeVerticalBold } from 'react-icons/pi';

export const CanvasEntityMenuButton = memo(() => {
  return (
    <MenuButton
      as={IconButton}
      aria-label="Layer menu"
      size="sm"
      icon={<PiDotsThreeVerticalBold />}
      onDoubleClick={stopPropagation} // double click expands the layer
    />
  );
});

CanvasEntityMenuButton.displayName = 'CanvasEntityMenuButton';
