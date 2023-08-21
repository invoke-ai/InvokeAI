import { Box } from '@chakra-ui/react';
import { memo } from 'react';

type Props = {
  isSelected: boolean;
  isHovered: boolean;
};
const SelectionOverlay = ({ isSelected, isHovered }: Props) => {
  return (
    <Box
      className="selection-box"
      sx={{
        position: 'absolute',
        top: 0,
        insetInlineEnd: 0,
        bottom: 0,
        insetInlineStart: 0,
        borderRadius: 'base',
        opacity: isSelected ? 1 : 0.7,
        transitionProperty: 'common',
        transitionDuration: '0.1s',
        pointerEvents: 'none',
        shadow: isSelected
          ? isHovered
            ? 'hoverSelected.light'
            : 'selected.light'
          : isHovered
          ? 'hoverUnselected.light'
          : undefined,
        _dark: {
          shadow: isSelected
            ? isHovered
              ? 'hoverSelected.dark'
              : 'selected.dark'
            : isHovered
            ? 'hoverUnselected.dark'
            : undefined,
        },
      }}
    />
  );
};

export default memo(SelectionOverlay);
