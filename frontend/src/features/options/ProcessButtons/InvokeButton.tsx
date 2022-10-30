import { useHotkeys } from 'react-hotkeys-hook';
import { generateImage } from '../../../app/socketio/actions';
import { useAppDispatch, useAppSelector } from '../../../app/store';
import IAIButton, {
  IAIButtonProps,
} from '../../../common/components/IAIButton';
import useCheckParameters from '../../../common/hooks/useCheckParameters';
import { activeTabNameSelector } from '../optionsSelectors';

export default function InvokeButton(props: Omit<IAIButtonProps, 'label'>) {
  const { ...rest } = props;
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

  return (
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
