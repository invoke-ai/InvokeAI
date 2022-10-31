import { useHotkeys } from 'react-hotkeys-hook';
import { FaPlay } from 'react-icons/fa';
import { readinessSelector } from '../../../app/selectors/readinessSelector';
import { generateImage } from '../../../app/socketio/actions';
import { useAppDispatch, useAppSelector } from '../../../app/store';
import IAIButton, {
  IAIButtonProps,
} from '../../../common/components/IAIButton';
import IAIIconButton from '../../../common/components/IAIIconButton';
import { activeTabNameSelector } from '../optionsSelectors';

interface InvokeButton extends Omit<IAIButtonProps, 'label'> {
  iconButton?: boolean;
}

export default function InvokeButton(props: InvokeButton) {
  const { iconButton = false, ...rest } = props;
  const dispatch = useAppDispatch();
  const isReady = useAppSelector(readinessSelector);
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
      icon={<FaPlay />}
      isDisabled={!isReady}
      onClick={handleClickGenerate}
      className="invoke-btn invoke"
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
