import { useAppDispatch } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { useSingleAndDoubleClick } from 'common/hooks/useSingleAndDoubleClick';
import { resetCanvasView } from 'features/canvas/store/canvasSlice';
import { getCanvasBaseLayer } from 'features/canvas/util/konvaInstanceProvider';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaCrosshairs } from 'react-icons/fa';

export default function UnifiedCanvasResetView() {
  const canvasBaseLayer = getCanvasBaseLayer();
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

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

  const handleClickResetCanvasView = useSingleAndDoubleClick(
    () => handleResetCanvasView(false),
    () => handleResetCanvasView(true)
  );

  const handleResetCanvasView = (shouldScaleTo1 = false) => {
    const canvasBaseLayer = getCanvasBaseLayer();
    if (!canvasBaseLayer) {
      return;
    }
    const clientRect = canvasBaseLayer.getClientRect({
      skipTransform: true,
    });
    dispatch(
      resetCanvasView({
        contentRect: clientRect,
        shouldScaleTo1,
      })
    );
  };
  return (
    <IAIIconButton
      aria-label={`${t('unifiedCanvas.resetView')} (R)`}
      tooltip={`${t('unifiedCanvas.resetView')} (R)`}
      icon={<FaCrosshairs />}
      onClick={handleClickResetCanvasView}
    />
  );
}
