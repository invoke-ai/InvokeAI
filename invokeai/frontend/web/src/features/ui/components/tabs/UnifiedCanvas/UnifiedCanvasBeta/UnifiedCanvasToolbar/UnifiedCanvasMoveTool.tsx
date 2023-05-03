import { RootState } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { setTool } from 'features/canvas/store/canvasSlice';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaArrowsAlt } from 'react-icons/fa';

export default function UnifiedCanvasMoveTool() {
  const tool = useAppSelector((state: RootState) => state.canvas.tool);
  const isStaging = useAppSelector(isStagingSelector);
  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  useHotkeys(
    ['v'],
    () => {
      handleSelectMoveTool();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    []
  );

  const handleSelectMoveTool = () => dispatch(setTool('move'));

  return (
    <IAIIconButton
      aria-label={`${t('unifiedCanvas.move')} (V)`}
      tooltip={`${t('unifiedCanvas.move')} (V)`}
      icon={<FaArrowsAlt />}
      isChecked={tool === 'move' || isStaging}
      onClick={handleSelectMoveTool}
    />
  );
}
