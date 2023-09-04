import {
  Box,
  Divider,
  Flex,
  ListItem,
  Text,
  UnorderedList,
} from '@chakra-ui/react';
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
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { memo, useCallback } from 'react';
import { useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { FaPlay } from 'react-icons/fa';
import { useBoardName } from 'services/api/hooks/useBoardName';

interface InvokeButton
  extends Omit<IAIButtonProps | IAIIconButtonProps, 'aria-label'> {
  asIconButton?: boolean;
}

export default function InvokeButton(props: InvokeButton) {
  const { asIconButton = false, sx, ...rest } = props;
  const dispatch = useAppDispatch();
  const { isReady, isProcessing } = useIsReadyToInvoke();
  const activeTabName = useAppSelector(activeTabNameSelector);

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
            sx={{
              position: 'absolute',
              bottom: '0',
              left: '0',
              right: '0',
              height: '100%',
              overflow: 'clip',
              borderRadius: 'base',
              ...sx,
            }}
            {...rest}
          >
            <ProgressBar />
          </Box>
        )}
        {asIconButton ? (
          <IAIIconButton
            aria-label={t('parameters.invoke')}
            type="submit"
            icon={<FaPlay />}
            isDisabled={!isReady}
            onClick={handleInvoke}
            tooltip={<InvokeButtonTooltipContent />}
            colorScheme="accent"
            isLoading={isProcessing}
            id="invoke-button"
            data-progress={isProcessing}
            sx={{
              w: 'full',
              flexGrow: 1,
              ...sx,
            }}
            {...rest}
          />
        ) : (
          <IAIButton
            tooltip={<InvokeButtonTooltipContent />}
            aria-label={t('parameters.invoke')}
            type="submit"
            data-progress={isProcessing}
            isDisabled={!isReady}
            onClick={handleInvoke}
            colorScheme="accent"
            id="invoke-button"
            leftIcon={isProcessing ? undefined : <FaPlay />}
            isLoading={isProcessing}
            loadingText={t('parameters.invoke')}
            sx={{
              w: 'full',
              flexGrow: 1,
              fontWeight: 700,
              ...sx,
            }}
            {...rest}
          >
            Invoke
          </IAIButton>
        )}
      </Box>
    </Box>
  );
}

const tooltipSelector = createSelector(
  [stateSelector],
  ({ gallery }) => {
    const { autoAddBoardId } = gallery;

    return {
      autoAddBoardId,
    };
  },
  defaultSelectorOptions
);

export const InvokeButtonTooltipContent = memo(() => {
  const { isReady, reasons } = useIsReadyToInvoke();
  const { autoAddBoardId } = useAppSelector(tooltipSelector);
  const autoAddBoardName = useBoardName(autoAddBoardId);

  return (
    <Flex flexDir="column" gap={1}>
      <Text fontWeight={600}>
        {isReady ? 'Ready to Invoke' : 'Unable to Invoke'}
      </Text>
      {reasons.length > 0 && (
        <UnorderedList>
          {reasons.map((reason, i) => (
            <ListItem key={`${reason}.${i}`}>
              <Text fontWeight={400}>{reason}</Text>
            </ListItem>
          ))}
        </UnorderedList>
      )}
      <Divider
        opacity={0.2}
        borderColor="base.50"
        _dark={{ borderColor: 'base.900' }}
      />
      <Text fontWeight={400} fontStyle="oblique 10deg">
        Adding images to{' '}
        <Text as="span" fontWeight={600}>
          {autoAddBoardName || 'Uncategorized'}
        </Text>
      </Text>
    </Flex>
  );
});

InvokeButtonTooltipContent.displayName = 'InvokeButtonTooltipContent';
