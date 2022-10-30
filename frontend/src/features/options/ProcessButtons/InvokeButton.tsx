import { useHotkeys } from 'react-hotkeys-hook';
import { BsImageFill } from 'react-icons/bs';
import { generateImage } from '../../../app/socketio/actions';
import { useAppDispatch, useAppSelector } from '../../../app/store';
import IAIButton, {
  IAIButtonProps,
} from '../../../common/components/IAIButton';
import IAIIconButton from '../../../common/components/IAIIconButton';
import useCheckParameters from '../../../common/hooks/useCheckParameters';
import { activeTabNameSelector } from '../optionsSelectors';

interface InvokeButton extends Omit<IAIButtonProps, 'label'> {
  iconButton?: boolean;
}

export default function InvokeButton(props: InvokeButton) {
  const { iconButton = false, ...rest } = props;
  const dispatch = useAppDispatch();
  const isReady = useCheckParameters();

  const activeTabName = useAppSelector(activeTabNameSelector);

  const handleClickGenerate = () => {
    dispatch(generateImage(activeTabName));
  };

  useHotkeys(
    'ctrl+enter, cmd+enter',
    () => {
      if (isReady) {
        dispatch(generateImage(activeTabName));
      }
    },
    [isReady, activeTabName]
  );

  return iconButton ? (
    <IAIIconButton
      aria-label="Invoke"
      type="submit"
      icon={<BsImageFill />}
      isDisabled={!isReady}
      onClick={handleClickGenerate}
      className="invoke-btn"
      tooltip="Invoke"
      tooltipPlacement="bottom"
      {...rest}
    />
  ) : (
    <IAIButton
      label="Invoke"
      aria-label="Invoke"
      type="submit"
      isDisabled={!isReady}
      onClick={handleClickGenerate}
      className="invoke-btn"
      {...rest}
    />
  );
}
