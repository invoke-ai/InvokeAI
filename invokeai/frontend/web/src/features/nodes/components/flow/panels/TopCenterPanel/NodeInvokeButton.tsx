import { Box } from '@chakra-ui/react';
import { userInvoked } from 'app/store/actions';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import IAIButton, { IAIButtonProps } from 'common/components/IAIButton';
import { IAIIconButtonProps } from 'common/components/IAIIconButton';
import { useIsReadyToInvoke } from 'common/hooks/useIsReadyToInvoke';
import { InvokeButtonTooltipContent } from 'features/parameters/components/ProcessButtons/InvokeButton';
import ProgressBar from 'features/system/components/ProgressBar';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaPlay } from 'react-icons/fa';

interface InvokeButton
  extends Omit<IAIButtonProps | IAIIconButtonProps, 'aria-label'> {}

const NodeInvokeButton = (props: InvokeButton) => {
  const { ...rest } = props;
  const dispatch = useAppDispatch();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const { isReady, isProcessing } = useIsReadyToInvoke();
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
        <IAIButton
          tooltip={<InvokeButtonTooltipContent />}
          aria-label={t('parameters.invoke')}
          type="submit"
          isDisabled={!isReady}
          onClick={handleInvoke}
          flexGrow={1}
          w="100%"
          colorScheme="accent"
          id="invoke-button"
          leftIcon={isProcessing ? undefined : <FaPlay />}
          fontWeight={700}
          isLoading={isProcessing}
          loadingText={t('parameters.invoke')}
          {...rest}
        >
          Invoke
        </IAIButton>
      </Box>
    </Box>
  );
};

export default memo(NodeInvokeButton);
