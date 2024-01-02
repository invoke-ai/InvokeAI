import type { ChakraProps } from '@chakra-ui/react';
import { Flex, Portal } from '@chakra-ui/react';
import { InvButtonGroup } from 'common/components/InvButtonGroup/InvButtonGroup';
import { InvIconButton } from 'common/components/InvIconButton/InvIconButton';
import CancelCurrentQueueItemButton from 'features/queue/components/CancelCurrentQueueItemButton';
import ClearQueueButton from 'features/queue/components/ClearQueueButton';
import { QueueButtonTooltip } from 'features/queue/components/QueueButtonTooltip';
import { useQueueBack } from 'features/queue/hooks/useQueueBack';
import type { RefObject } from 'react';
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { FaSlidersH } from 'react-icons/fa';
import { IoSparkles } from 'react-icons/io5';
import type { ImperativePanelHandle } from 'react-resizable-panels';

const floatingButtonStyles: ChakraProps['sx'] = {
  borderStartRadius: 0,
  flexGrow: 1,
};

type Props = {
  isSidePanelCollapsed: boolean;
  sidePanelRef: RefObject<ImperativePanelHandle>;
};

const FloatingSidePanelButtons = ({
  isSidePanelCollapsed,
  sidePanelRef,
}: Props) => {
  const { t } = useTranslation();
  const { queueBack, isLoading, isDisabled } = useQueueBack();

  const handleShowSidePanel = useCallback(() => {
    sidePanelRef.current?.expand();
  }, [sidePanelRef]);

  if (!isSidePanelCollapsed) {
    return null;
  }

  return (
    <Portal>
      <Flex
        pos="absolute"
        transform="translate(0, -50%)"
        minW={8}
        top="50%"
        insetInlineStart="5.13rem"
        direction="column"
        gap={2}
        h={48}
      >
        <InvButtonGroup orientation="vertical" flexGrow={3}>
          <InvIconButton
            tooltip={t('parameters.showOptionsPanel')}
            aria-label={t('parameters.showOptionsPanel')}
            onClick={handleShowSidePanel}
            sx={floatingButtonStyles}
            icon={<FaSlidersH />}
          />
          <InvIconButton
            aria-label={t('queue.queueBack')}
            pos="absolute"
            insetInlineStart={0}
            onClick={queueBack}
            isLoading={isLoading}
            isDisabled={isDisabled}
            icon={<IoSparkles />}
            variant="solid"
            colorScheme="yellow"
            tooltip={<QueueButtonTooltip />}
            sx={floatingButtonStyles}
          />
          <CancelCurrentQueueItemButton
            asIconButton
            sx={floatingButtonStyles}
          />
        </InvButtonGroup>
        <ClearQueueButton asIconButton sx={floatingButtonStyles} />
      </Flex>
    </Portal>
  );
};

export default memo(FloatingSidePanelButtons);
