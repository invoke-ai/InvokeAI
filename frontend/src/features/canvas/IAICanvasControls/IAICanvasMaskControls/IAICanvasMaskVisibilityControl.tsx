import { useHotkeys } from 'react-hotkeys-hook';
import { BiHide, BiShow } from 'react-icons/bi';
import { createSelector } from 'reselect';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import {
  currentCanvasSelector,
  GenericCanvasState,
  setShouldShowMask,
} from 'features/canvas/canvasSlice';

import _ from 'lodash';

const canvasMaskVisibilitySelector = createSelector(
  [currentCanvasSelector, activeTabNameSelector],
  (currentCanvas: GenericCanvasState, activeTabName) => {
    const { shouldShowMask } = currentCanvas;

    return { shouldShowMask, activeTabName };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function IAICanvasMaskVisibilityControl() {
  const dispatch = useAppDispatch();

  const { shouldShowMask, activeTabName } = useAppSelector(
    canvasMaskVisibilitySelector
  );

  const handleToggleShouldShowMask = () =>
    dispatch(setShouldShowMask(!shouldShowMask));
  // Hotkeys
  // Show/hide mask
  useHotkeys(
    'h',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleToggleShouldShowMask();
    },
    {
      enabled: activeTabName === 'inpainting' || activeTabName == 'outpainting',
    },
    [activeTabName, shouldShowMask]
  );
  return (
    <IAIIconButton
      aria-label="Hide Mask (H)"
      tooltip="Hide Mask (H)"
      data-alert={!shouldShowMask}
      icon={shouldShowMask ? <BiShow size={22} /> : <BiHide size={22} />}
      onClick={handleToggleShouldShowMask}
    />
  );
}
