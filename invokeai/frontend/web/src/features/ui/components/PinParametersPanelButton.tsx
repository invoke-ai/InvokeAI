import { Tooltip } from '@chakra-ui/react';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { requestCanvasRescale } from 'features/canvas/store/thunks/requestCanvasScale';
import { useTranslation } from 'react-i18next';
import { BsPinAngle, BsPinAngleFill } from 'react-icons/bs';
import { setShouldPinParametersPanel } from '../store/uiSlice';

type PinParametersPanelButtonProps = Omit<IAIIconButtonProps, 'aria-label'>;

const PinParametersPanelButton = (props: PinParametersPanelButtonProps) => {
  const { sx } = props;
  const dispatch = useAppDispatch();
  const shouldPinParametersPanel = useAppSelector(
    (state) => state.ui.shouldPinParametersPanel
  );

  const { t } = useTranslation();

  const handleClickPinOptionsPanel = () => {
    dispatch(setShouldPinParametersPanel(!shouldPinParametersPanel));
    dispatch(requestCanvasRescale());
  };

  return (
    <Tooltip label={t('common.pinOptionsPanel')}>
      <IAIIconButton
        {...props}
        aria-label={t('common.pinOptionsPanel')}
        onClick={handleClickPinOptionsPanel}
        icon={shouldPinParametersPanel ? <BsPinAngleFill /> : <BsPinAngle />}
        variant="ghost"
        size="sm"
        px={{ base: 10, xl: 0 }}
        sx={{
          color: 'base.700',
          _hover: {
            color: 'base.550',
          },
          _active: {
            color: 'base.500',
          },
          ...sx,
        }}
      />
    </Tooltip>
  );
};

export default PinParametersPanelButton;
