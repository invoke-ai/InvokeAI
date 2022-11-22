import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store';
import { canvasSelector } from 'features/canvas/store/canvasSelectors';
import React from 'react';
import _ from 'lodash';

const cursorPositionSelector = createSelector(
  [canvasSelector],
  (canvas) => {
    const { cursorPosition } = canvas;

    const { cursorX, cursorY } = cursorPosition
      ? { cursorX: cursorPosition.x, cursorY: cursorPosition.y }
      : { cursorX: -1, cursorY: -1 };

    return {
      cursorCoordinatesString: `(${cursorX}, ${cursorY})`,
    };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function IAICanvasStatusTextCursorPos() {
  const { cursorCoordinatesString } = useAppSelector(cursorPositionSelector);

  return <div>{`Cursor Position: ${cursorCoordinatesString}`}</div>;
}
