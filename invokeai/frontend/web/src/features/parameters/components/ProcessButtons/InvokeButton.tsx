import { Box, ChakraProps, Tooltip } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { userInvoked } from 'app/store/actions';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton, { IAIButtonProps } from 'common/components/IAIButton';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { useIsReadyToInvoke } from 'common/hooks/useIsReadyToInvoke';
import { clampSymmetrySteps } from 'features/parameters/store/generationSlice';
import ProgressBar from 'features/system/components/ProgressBar';
import { selectIsBusy } from 'features/system/store/systemSelectors';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaPlay } from 'react-icons/fa';
import { useBoardName } from 'services/api/hooks/useBoardName';

const IN_PROGRESS_STYLES: ChakraProps['sx'] = {
  _disabled: {
    bg: 'none',
    color: 'base.600',
    cursor: 'not-allowed',
    _hover: {
      color: 'base.600',
      bg: 'none',
    },
  },
};

const selector = createSelector(
  [stateSelector, activeTabNameSelector, selectIsBusy],
  ({ gallery }, activeTabName, isBusy) => {
    const { autoAddBoardId } = gallery;

    return {
      isBusy,
      autoAddBoardId,
      activeTabName,
    };
  },
  defaultSelectorOptions
);

interface InvokeButton
  extends Omit<IAIButtonProps | IAIIconButtonProps, 'aria-label'> {
  iconButton?: boolean;
}

export default function InvokeButton(props: InvokeButton) {
  const { iconButton = false, ...rest } = props;
  const dispatch = useAppDispatch();
  const isReady = useIsReadyToInvoke();
  const { isBusy, autoAddBoardId, activeTabName } = useAppSelector(selector);
  const autoAddBoardName = useBoardName(autoAddBoardId);

  const handleInvoke = useCallback(() => {
    dispatch(clampSymmetrySteps());
    dispatch(userInvoked(activeTabName));
  }, [dispatch, activeTabName]);

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
            {...rest}
          >
            <ProgressBar />
          </Box>
        )}
        <Tooltip
          placement="top"
          hasArrow
          openDelay={500}
          label={
            autoAddBoardId ? `Auto-Adding to ${autoAddBoardName}` : undefined
          }
        >
          {iconButton ? (
            <IAIIconButton
              aria-label={t('parameters.invoke')}
              type="submit"
              icon={<FaPlay />}
              isDisabled={!isReady || isBusy}
              onClick={handleInvoke}
              tooltip={t('parameters.invoke')}
              tooltipProps={{ placement: 'top' }}
              colorScheme="accent"
              id="invoke-button"
              {...rest}
              sx={{
                w: 'full',
                flexGrow: 1,
                ...(isBusy ? IN_PROGRESS_STYLES : {}),
              }}
            />
          ) : (
            <IAIButton
              aria-label={t('parameters.invoke')}
              type="submit"
              isDisabled={!isReady || isBusy}
              onClick={handleInvoke}
              colorScheme="accent"
              id="invoke-button"
              {...rest}
              sx={{
                w: 'full',
                flexGrow: 1,
                fontWeight: 700,
                ...(isBusy ? IN_PROGRESS_STYLES : {}),
              }}
            >
              Invoke
            </IAIButton>
          )}
        </Tooltip>
      </Box>
    </Box>
  );
}
