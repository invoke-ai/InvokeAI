import type { SystemStyleObject } from '@invoke-ai/ui-library';
import { ButtonGroup, Flex, Icon, IconButton, Portal, spinAnimation, useDisclosure } from '@invoke-ai/ui-library';
import CancelCurrentQueueItemIconButton from 'features/queue/components/CancelCurrentQueueItemIconButton';
import ClearQueueConfirmationAlertDialog from 'features/queue/components/ClearQueueConfirmationAlertDialog';
import { ClearAllQueueIconButton } from 'features/queue/components/ClearQueueIconButton';
import { QueueButtonTooltip } from 'features/queue/components/QueueButtonTooltip';
import { useQueueBack } from 'features/queue/hooks/useQueueBack';
import type { UsePanelReturn } from 'features/ui/hooks/usePanel';
import { memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCircleNotchBold, PiSlidersHorizontalBold } from 'react-icons/pi';
import { RiSparklingFill } from 'react-icons/ri';
import { useGetQueueStatusQuery } from 'services/api/endpoints/queue';

const floatingButtonStyles: SystemStyleObject = {
  borderStartRadius: 0,
  flexGrow: 1,
};

type Props = {
  panelApi: UsePanelReturn;
};

const FloatingSidePanelButtons = (props: Props) => {
  const { t } = useTranslation();
  const { queueBack, isLoading, isDisabled } = useQueueBack();
  const { data: queueStatus } = useGetQueueStatusQuery();

  const queueButtonIcon = useMemo(
    () =>
      !isDisabled && queueStatus?.processor.is_processing ? (
        <Icon boxSize={6} as={PiCircleNotchBold} animation={spinAnimation} />
      ) : (
        <RiSparklingFill size="16px" />
      ),
    [isDisabled, queueStatus?.processor.is_processing]
  );

  const disclosure = useDisclosure();

  if (!props.panelApi.isCollapsed) {
    return null;
  }

  return (
    <Portal>
      <Flex
        pos="absolute"
        transform="translate(0, -50%)"
        minW={8}
        top="50%"
        insetInlineStart="63px"
        direction="column"
        gap={2}
        h={48}
      >
        <ButtonGroup orientation="vertical" flexGrow={3}>
          <IconButton
            tooltip={t('accessibility.showOptionsPanel')}
            aria-label={t('accessibility.showOptionsPanel')}
            onClick={props.panelApi.expand}
            sx={floatingButtonStyles}
            icon={<PiSlidersHorizontalBold size="16px" />}
          />
          <IconButton
            aria-label={t('queue.queueBack')}
            onClick={queueBack}
            isLoading={isLoading}
            isDisabled={isDisabled}
            icon={queueButtonIcon}
            colorScheme="invokeYellow"
            tooltip={<QueueButtonTooltip />}
            sx={floatingButtonStyles}
          />
          <CancelCurrentQueueItemIconButton sx={floatingButtonStyles} />
        </ButtonGroup>
        <ClearAllQueueIconButton sx={floatingButtonStyles} onOpen={disclosure.onOpen} />
        <ClearQueueConfirmationAlertDialog disclosure={disclosure} />
      </Flex>
    </Portal>
  );
};

export default memo(FloatingSidePanelButtons);
