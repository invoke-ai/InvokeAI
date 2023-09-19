import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { canvasMerged } from 'features/canvas/store/actions';
import { isStagingSelector } from 'features/canvas/store/canvasSelectors';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaLayerGroup } from 'react-icons/fa';
export default function UnifiedCanvasMergeVisible() {
  const dispatch = useAppDispatch();
  const { t } = useTranslation();
  const isStaging = useAppSelector(isStagingSelector);

  useHotkeys(
    ['shift+m'],
    () => {
      handleMergeVisible();
    },
    {
      enabled: () => !isStaging,
      preventDefault: true,
    },
    []
  );

  const handleMergeVisible = () => {
    dispatch(canvasMerged());
  };
  return (
    <IAIIconButton
      aria-label={`${t('unifiedCanvas.mergeVisible')} (Shift+M)`}
      tooltip={`${t('unifiedCanvas.mergeVisible')} (Shift+M)`}
      icon={<FaLayerGroup />}
      onClick={handleMergeVisible}
      isDisabled={isStaging}
    />
  );
}
