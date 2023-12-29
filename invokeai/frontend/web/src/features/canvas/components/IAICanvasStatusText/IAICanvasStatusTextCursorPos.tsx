import { Box } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import roundToHundreth from 'features/canvas/util/roundToHundreth';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const cursorPositionSelector = createMemoizedSelector(
  [stateSelector],
  ({ canvas }) => {
    const { cursorPosition } = canvas;

    const { cursorX, cursorY } = cursorPosition
      ? { cursorX: cursorPosition.x, cursorY: cursorPosition.y }
      : { cursorX: -1, cursorY: -1 };

    return {
      cursorCoordinatesString: `(${roundToHundreth(cursorX)}, ${roundToHundreth(
        cursorY
      )})`,
    };
  }
);

const IAICanvasStatusTextCursorPos = () => {
  const { cursorCoordinatesString } = useAppSelector(cursorPositionSelector);
  const { t } = useTranslation();

  return (
    <Box>{`${t(
      'unifiedCanvas.cursorPosition'
    )}: ${cursorCoordinatesString}`}</Box>
  );
};

export default memo(IAICanvasStatusTextCursorPos);
