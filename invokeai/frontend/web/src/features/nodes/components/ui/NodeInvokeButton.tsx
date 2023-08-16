import { Box } from '@chakra-ui/react';
import { userInvoked } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton, { IAIButtonProps } from 'common/components/IAIButton';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { selectIsReadyNodes } from 'features/nodes/store/selectors';
import ProgressBar from 'features/system/components/ProgressBar';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaPlay } from 'react-icons/fa';

interface InvokeButton
  extends Omit<IAIButtonProps | IAIIconButtonProps, 'aria-label'> {
  iconButton?: boolean;
}

const NodeInvokeButton = (props: InvokeButton) => {
  const { iconButton = false, ...rest } = props;
  const dispatch = useAppDispatch();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const isReady = useAppSelector(selectIsReadyNodes);
  const handleInvoke = useCallback(() => {
    dispatch(userInvoked('nodes'));
  }, [dispatch]);

  const { t } = useTranslation();

  useHotkeys(
    ['ctrl+enter', 'meta+enter'],
    handleInvoke,
    {
      enabled: () => isReady,
      preventDefault: true,
      enableOnFormTags: ['input', 'textarea', 'select'],
    },
    [isReady, activeTabName]
  );

  return (
    <Box style={{ flexGrow: 4 }} position="relative">
      <Box style={{ position: 'relative' }}>
        {!isReady && (
          <Box
            borderRadius="base"
            style={{
              position: 'absolute',
              bottom: '0',
              left: '0',
              right: '0',
              height: '100%',
              overflow: 'clip',
            }}
          >
            <ProgressBar />
          </Box>
        )}
        {iconButton ? (
          <IAIIconButton
            aria-label={t('parameters.invoke')}
            type="submit"
            icon={<FaPlay />}
            isDisabled={!isReady}
            onClick={handleInvoke}
            flexGrow={1}
            w="100%"
            tooltip={t('parameters.invoke')}
            tooltipProps={{ placement: 'bottom' }}
            colorScheme="accent"
            id="invoke-button"
            {...rest}
          />
        ) : (
          <IAIButton
            aria-label={t('parameters.invoke')}
            type="submit"
            isDisabled={!isReady}
            onClick={handleInvoke}
            flexGrow={1}
            w="100%"
            colorScheme="accent"
            id="invoke-button"
            fontWeight={700}
            {...rest}
          >
            Invoke
          </IAIButton>
        )}
      </Box>
    </Box>
  );
};

export default memo(NodeInvokeButton);
