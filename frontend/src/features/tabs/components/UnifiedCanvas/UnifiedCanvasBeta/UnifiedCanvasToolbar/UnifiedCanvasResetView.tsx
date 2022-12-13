import { useAppDispatch } from 'app/store';
import IAIIconButton from 'common/components/IAIIconButton';
import { resetCanvasView } from 'features/canvas/store/canvasSlice';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import React from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaCrosshairs } from 'react-icons/fa';

export default function UnifiedCanvasResetView() {
  const canvasBaseLayer = getCanvasBaseLayer();
  const dispatch = useAppDispatch();

  useHotkeys(
    ['r'],
    () => {
      handleResetCanvasView();
    },
    {
      enabled: () => true,
      preventDefault: true,
    },
    [canvasBaseLayer]
  );

  const handleResetCanvasView = () => {
    const canvasBaseLayer = getCanvasBaseLayer();
    if (!canvasBaseLayer) return;
    const clientRect = canvasBaseLayer.getClientRect({
      skipTransform: true,
    });
    dispatch(
      resetCanvasView({
        contentRect: clientRect,
      })
    );
  };
  return (
    <IAIIconButton
      aria-label="Reset View (R)"
      tooltip="Reset View (R)"
      icon={<FaCrosshairs />}
      onClick={handleResetCanvasView}
    />
  );
}
