import { Box } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { $cursorPosition } from 'features/canvas/store/canvasNanostore';
import roundToHundreth from 'features/canvas/util/roundToHundreth';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

const IAICanvasStatusTextCursorPos = () => {
  const { t } = useTranslation();
  const cursorPosition = useStore($cursorPosition);
  const cursorCoordinatesString = useMemo(() => {
    const x = cursorPosition?.x ?? -1;
    const y = cursorPosition?.y ?? -1;
    return `(${roundToHundreth(x)}, ${roundToHundreth(y)})`;
  }, [cursorPosition?.x, cursorPosition?.y]);

  return <Box>{`${t('unifiedCanvas.cursorPosition')}: ${cursorCoordinatesString}`}</Box>;
};

export default memo(IAICanvasStatusTextCursorPos);
