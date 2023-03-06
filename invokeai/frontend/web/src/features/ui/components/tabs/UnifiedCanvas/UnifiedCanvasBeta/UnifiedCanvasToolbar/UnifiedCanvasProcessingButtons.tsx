import { Flex } from '@chakra-ui/layout';
import { RootState } from 'app/store';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import InvokeButton from 'features/parameters/components/ProcessButtons/InvokeButton';
import { setShouldShowParametersPanel } from 'features/ui/store/uiSlice';
import { useTranslation } from 'react-i18next';
import { FaSlidersH } from 'react-icons/fa';

export default function UnifiedCanvasProcessingButtons() {
  const shouldPinParametersPanel = useAppSelector(
    (state: RootState) => state.ui.shouldPinParametersPanel
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleShowOptionsPanel = () => {
    dispatch(setShouldShowParametersPanel(true));
    if (shouldPinParametersPanel) {
      setTimeout(() => dispatch(setDoesCanvasNeedScaling(true)), 400);
    }
  };

  return (
    <Flex flexDirection="column" gap={2}>
      <IAIIconButton
        tooltip={`${t('parameters.showOptionsPanel')} (O)`}
        tooltipProps={{ placement: 'top' }}
        aria-label={t('parameters.showOptionsPanel')}
        onClick={handleShowOptionsPanel}
      >
        <FaSlidersH />
      </IAIIconButton>
      <Flex>
        <InvokeButton iconButton />
      </Flex>
      <Flex>
        <CancelButton width="100%" height="40px" btnGroupWidth="100%" />
      </Flex>
    </Flex>
  );
}
