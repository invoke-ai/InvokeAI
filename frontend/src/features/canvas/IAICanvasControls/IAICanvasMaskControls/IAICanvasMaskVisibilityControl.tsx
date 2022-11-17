import { useHotkeys } from 'react-hotkeys-hook';
import { BiHide, BiShow } from 'react-icons/bi';
import { createSelector } from 'reselect';
import { useAppDispatch, useAppSelector } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { activeTabNameSelector } from 'features/options/optionsSelectors';
import { canvasSelector, setIsMaskEnabled } from 'features/canvas/canvasSlice';

import _ from 'lodash';

const canvasMaskVisibilitySelector = createSelector(
  [canvasSelector, activeTabNameSelector],
  (canvas, activeTabName) => {
    const { isMaskEnabled } = canvas;

    return { isMaskEnabled, activeTabName };
  },
  {
    memoizeOptions: {
      resultEqualityCheck: _.isEqual,
    },
  }
);

export default function IAICanvasMaskVisibilityControl() {
  const dispatch = useAppDispatch();

  const { isMaskEnabled, activeTabName } = useAppSelector(
    canvasMaskVisibilitySelector
  );

  const handleToggleShouldShowMask = () =>
    dispatch(setIsMaskEnabled(!isMaskEnabled));
  // Hotkeys
  // Show/hide mask
  useHotkeys(
    'h',
    (e: KeyboardEvent) => {
      e.preventDefault();
      handleToggleShouldShowMask();
    },
    {
      enabled: activeTabName === 'unifiedCanvas',
    },
    [activeTabName, isMaskEnabled]
  );
  return (
    <IAIIconButton
      aria-label="Hide Mask (H)"
      tooltip="Hide Mask (H)"
      data-alert={!isMaskEnabled}
      icon={isMaskEnabled ? <BiShow size={22} /> : <BiHide size={22} />}
      onClick={handleToggleShouldShowMask}
    />
  );
}
