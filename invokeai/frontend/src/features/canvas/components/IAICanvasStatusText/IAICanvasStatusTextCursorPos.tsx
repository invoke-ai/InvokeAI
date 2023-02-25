import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/storeHooks';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import roundToHundreth from 'features/canvas/util/roundToHundreth';
import { isEqual } from 'lodash';

import { useTranslation } from 'react-i18next';

const cursorPositionSelector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { cursorPosition } = canvas;

    const { cursorX, cursorY } = cursorPosition
      ? { cursorX: cursorPosition.x, cursorY: cursorPosition.y }
      : { cursorX: -1, cursorY: -1 };

    return {
      cursorCoordinatesString: `(${roundToHundreth(cursorX)}, ${roundToHundreth(
        cursorY
      )})`,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: isEqual,
    },
  }
);

export default function IAICanvasStatusTextCursorPos() {
  const { cursorCoordinatesString } = useAppSelector(cursorPositionSelector);
  const { t } = useTranslation();

  return (
    <div>{`${t(
      'unifiedCanvas.cursorPosition'
    )}: ${cursorCoordinatesString}`}</div>
  );
}
