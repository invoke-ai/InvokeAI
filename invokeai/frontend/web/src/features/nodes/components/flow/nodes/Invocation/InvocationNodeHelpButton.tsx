import { Icon, useDisclosure } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { PiQuestionBold } from 'react-icons/pi';

import { InvocationNodeHelpModal } from './InvocationNodeHelpModal';

interface Props {
  nodeId: string;
}

export const InvocationNodeHelpButton = memo(({ nodeId: _nodeId }: Props) => {
  const { isOpen, onOpen, onClose } = useDisclosure();

  return (
    <>
      <Icon
        as={PiQuestionBold}
        display="block"
        boxSize={4}
        w={8}
        cursor="pointer"
        onClick={onOpen}
        _hover={{ color: 'base.300' }}
        aria-label="Help"
      />
      <InvocationNodeHelpModal isOpen={isOpen} onClose={onClose} />
    </>
  );
});

InvocationNodeHelpButton.displayName = 'InvocationNodeHelpButton';
