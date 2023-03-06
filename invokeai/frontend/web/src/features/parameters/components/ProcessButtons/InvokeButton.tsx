import { readinessSelector } from 'app/selectors/readinessSelector';
import { generateImage } from 'app/socketio/actions';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import IAIButton, { IAIButtonProps } from 'common/components/IAIButton';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaPlay } from 'react-icons/fa';

interface InvokeButton
  extends Omit<IAIButtonProps | IAIIconButtonProps, 'aria-label'> {
  iconButton?: boolean;
}

export default function InvokeButton(props: InvokeButton) {
  const { iconButton = false, ...rest } = props;
  const dispatch = useAppDispatch();
  const { isReady } = useAppSelector(readinessSelector);
  const activeTabName = useAppSelector(activeTabNameSelector);

  const handleClickGenerate = () => {
    dispatch(generateImage(activeTabName));
  };

  const { t } = useTranslation();

  useHotkeys(
    ['ctrl+enter', 'meta+enter'],
    () => {
      dispatch(generateImage(activeTabName));
    },
    {
      enabled: () => isReady,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    [isReady, activeTabName]
  );

  return (
    <div style={{ flexGrow: 4 }}>
      {iconButton ? (
        <IAIIconButton
          aria-label={t('parameters.invoke')}
          type="submit"
          icon={<FaPlay />}
          isDisabled={!isReady}
          onClick={handleClickGenerate}
          className="invoke-btn"
          tooltip={t('parameters.invoke')}
          tooltipProps={{ placement: 'bottom' }}
          {...rest}
        />
      ) : (
        <IAIButton
          aria-label={t('parameters.invoke')}
          type="submit"
          isDisabled={!isReady}
          onClick={handleClickGenerate}
          className="invoke-btn"
          {...rest}
        >
          Invoke
        </IAIButton>
      )}
    </div>
  );
}
