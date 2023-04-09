import { Flex } from '@chakra-ui/react';
import LoraManager from './LoraManager/LoraManager';
import TextualInversionManager from './TextualInversionManager/TextualInversionManager';

export default function PromptExtras() {
  return (
    <Flex flexDir="column" rowGap={2}>
      <LoraManager />
      <TextualInversionManager />
    </Flex>
  );
}
