import { ListItem, UnorderedList } from '@chakra-ui/react';
import { useHotkeys } from 'react-hotkeys-hook';
import { FaPlay } from 'react-icons/fa';
import { readinessSelector } from '../../../app/selectors/readinessSelector';
import { generateImage } from '../../../app/socketio/actions';
import { useAppDispatch, useAppSelector } from '../../../app/store';
import IAIButton, {
  IAIButtonProps,
} from '../../../common/components/IAIButton';
import IAIIconButton from '../../../common/components/IAIIconButton';
import IAIPopover from '../../../common/components/IAIPopover';
import { activeTabNameSelector } from '../optionsSelectors';

interface InvokeButton extends Omit<IAIButtonProps, 'label'> {
  iconButton?: boolean;
}

export default function InvokeButton(props: InvokeButton) {
  const { iconButton = false, ...rest } = props;
  const dispatch = useAppDispatch();
  const { isReady, reasonsWhyNotReady } = useAppSelector(readinessSelector);
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

  const buttonComponent = iconButton ? (
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

  return isReady ? (
    buttonComponent
  ) : (
    <IAIPopover
      trigger="hover"
      triggerContainerProps={{ style: { flexGrow: 4 } }}
      triggerComponent={buttonComponent}
    >
      {reasonsWhyNotReady && (
        <UnorderedList>
          {reasonsWhyNotReady.map((reason, i) => (
            <ListItem key={i}>{reason}</ListItem>
          ))}
        </UnorderedList>
      )}
    </IAIPopover>
  );
}
