import { Flex } from '@chakra-ui/layout';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton from 'common/components/IAIIconButton';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import InvokeButton from 'features/parameters/components/ProcessButtons/InvokeButton';
import { setShouldShowParametersPanel } from 'features/ui/store/uiSlice';
import { useTranslation } from 'react-i18next';
import { FaSlidersH } from 'react-icons/fa';

export default function UnifiedCanvasProcessingButtons() {
  const shouldPinParametersPanel = useAppSelector(
    (state) => state.ui.shouldPinParametersPanel
  );
  const shouldShowParametersPanel = useAppSelector(
    (state) => state.ui.shouldShowParametersPanel
  );

  const dispatch = useAppDispatch();
  const { t } = useTranslation();

  const handleShowOptionsPanel = () => {
    dispatch(setShouldShowParametersPanel(true));
    shouldPinParametersPanel && dispatch(requestCanvasRescale());
  };

  return !shouldPinParametersPanel || !shouldShowParametersPanel ? (
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
  ) : null;
}
