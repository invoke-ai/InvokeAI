import {
  Box,
  Divider,
  Flex,
  ListItem,
  Text,
  UnorderedList,
} from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { enqueueRequested } from 'app/store/actions';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton, { IAIButtonProps } from 'common/components/IAIButton';
import IAIIconButton, {
  IAIIconButtonProps,
} from 'common/components/IAIIconButton';
import { useIsReadyToEnqueue } from 'common/hooks/useIsReadyToEnqueue';
import { clampSymmetrySteps } from 'features/parameters/store/generationSlice';
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
  const { isReady, isProcessing } = useIsReadyToEnqueue();
  const activeTabName = useAppSelector(activeTabNameSelector);
  const handleInvoke = useCallback(() => {
    dispatch(clampSymmetrySteps());
    dispatch(enqueueRequested({ tabName: activeTabName, prepend: true }));
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
        {asIconButton ? (
          <IAIIconButton
            aria-label={t('parameters.invoke.invoke')}
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
            aria-label={t('parameters.invoke.invoke')}
            type="submit"
            onClick={handleInvoke}
            colorScheme="accent"
            id="invoke-button"
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
  const { isReady, reasons } = useIsReadyToEnqueue();
  const { autoAddBoardId } = useAppSelector(tooltipSelector);
  const autoAddBoardName = useBoardName(autoAddBoardId);
  const { t } = useTranslation();

  return (
    <Flex flexDir="column" gap={1}>
      <Text fontWeight={600}>
        {isReady
          ? t('parameters.invoke.readyToInvoke')
          : t('parameters.invoke.unableToInvoke')}
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
        {t('parameters.invoke.addingImagesTo')}{' '}
        <Text as="span" fontWeight={600}>
          {autoAddBoardName || 'Uncategorized'}
        </Text>
      </Text>
    </Flex>
  );
});

InvokeButtonTooltipContent.displayName = 'InvokeButtonTooltipContent';
