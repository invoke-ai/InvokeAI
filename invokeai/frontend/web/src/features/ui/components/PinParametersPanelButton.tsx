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
    <IAIIconButton
      {...props}
      tooltip={t('common.pinOptionsPanel')}
      aria-label={t('common.pinOptionsPanel')}
      onClick={handleClickPinOptionsPanel}
      icon={shouldPinParametersPanel ? <BsPinAngleFill /> : <BsPinAngle />}
      variant="ghost"
      size="sm"
      sx={{
        color: 'base.500',
        _hover: {
          color: 'base.600',
        },
        _active: {
          color: 'base.700',
        },
        _dark: {
          color: 'base.500',
          _hover: {
            color: 'base.400',
          },
          _active: {
            color: 'base.300',
          },
        },
        ...sx,
      }}
    />
  );
};

export default PinParametersPanelButton;
