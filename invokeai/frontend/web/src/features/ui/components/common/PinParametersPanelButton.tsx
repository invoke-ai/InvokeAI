import { Box, Icon, Tooltip } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { setDoesCanvasNeedScaling } from 'features/canvas/store/canvasSlice';
import { useTranslation } from 'react-i18next';
import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';
import { setShouldPinParametersPanel } from '../../store/uiSlice';

const PinParametersPanelButton = () => {
  const dispatch = useAppDispatch();
  const shouldPinParametersPanel = useAppSelector(
    (state) => state.ui.shouldPinParametersPanel
  );

  const { t } = useTranslation();

  const handleClickPinOptionsPanel = () => {
    dispatch(setShouldPinParametersPanel(!shouldPinParametersPanel));
    dispatch(setDoesCanvasNeedScaling(true));
  };

  return (
    <Tooltip label={t('common.pinOptionsPanel')}>
      <IAIIconButton
        aria-label={t('common.pinOptionsPanel')}
        opacity={0.2}
        onClick={handleClickPinOptionsPanel}
        icon={shouldPinParametersPanel ? <BsPinAngleFill /> : <BsPinAngle />}
        variant="unstyled"
        size="sm"
        padding={2}
        sx={{
          position: 'absolute',
          top: 1,
          insetInlineEnd: 1,
        }}
      />
    </Tooltip>
  );
};

export default PinParametersPanelButton;
