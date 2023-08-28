import { ChakraProps, Flex, Portal } from '@chakra-ui/react';
import IAIIconButton from 'common/components/IAIIconButton';
import CancelButton from 'features/parameters/components/ProcessButtons/CancelButton';
import InvokeButton from 'features/parameters/components/ProcessButtons/InvokeButton';
import { RefObject, memo } from 'react';
import { useTranslation } from 'react-i18next';

import { FaSlidersH } from 'react-icons/fa';
import { ImperativePanelHandle } from 'react-resizable-panels';

const floatingButtonStyles: ChakraProps['sx'] = {
  borderStartStartRadius: 0,
  borderEndStartRadius: 0,
  shadow: '2xl',
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
      >
        <IAIIconButton
          tooltip="Show Side Panel (O, T)"
          aria-label={t('common.showOptionsPanel')}
          onClick={handleShowSidePanel}
          sx={floatingButtonStyles}
          icon={<FaSlidersH />}
        />
        <InvokeButton asIconButton sx={floatingButtonStyles} />
        <CancelButton sx={floatingButtonStyles} asIconButton />
      </Flex>
    </Portal>
  );
};

export default memo(FloatingSidePanelButtons);
