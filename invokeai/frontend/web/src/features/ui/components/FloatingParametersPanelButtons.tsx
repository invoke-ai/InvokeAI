import { ButtonGroup, ChakraProps, Flex, Portal } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import CancelCurrentQueueItemButton from 'features/queue/components/CancelCurrentQueueItemButton';
import ClearQueueButton from 'features/queue/components/ClearQueueButton';
import QueueBackButton from 'features/queue/components/QueueBackButton';
import { RefObject, memo } from 'react';
import { useTranslation } from 'react-i18next';

import { FaSlidersH } from 'react-icons/fa';
import { ImperativePanelHandle } from 'react-resizable-panels';

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

  const handleShowSidePanel = () => {
    sidePanelRef.current?.expand();
  };

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
        <ButtonGroup isAttached orientation="vertical" flexGrow={3}>
          <IAIIconButton
            tooltip="Show Side Panel (O, T)"
            aria-label={t('common.showOptionsPanel')}
            onClick={handleShowSidePanel}
            sx={floatingButtonStyles}
            icon={<FaSlidersH />}
          />
          <QueueBackButton asIconButton sx={floatingButtonStyles} />
          <CancelCurrentQueueItemButton
            asIconButton
            sx={floatingButtonStyles}
          />
        </ButtonGroup>
        <ClearQueueButton asIconButton sx={floatingButtonStyles} />
      </Flex>
    </Portal>
  );
};

export default memo(FloatingSidePanelButtons);
