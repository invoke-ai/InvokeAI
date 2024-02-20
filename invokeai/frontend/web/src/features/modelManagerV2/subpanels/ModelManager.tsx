import {
  Box,
  Button,
  Flex,
  Heading,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Spacer,
} from '@invoke-ai/ui-library';
import { t } from 'i18next';
import { PiXBold } from 'react-icons/pi';
import { SyncModelsIconButton } from '../../modelManager/components/SyncModels/SyncModelsIconButton';

export const ModelManager = () => {
  return (
    <Box layerStyle="first" p={3} borderRadius="base" w="full" h="full">
      <Flex w="full" p={3} justifyContent="space-between" alignItems="center">
        <Flex gap={2}>
          <Heading fontSize="xl">Model Manager</Heading>
          <SyncModelsIconButton />
        </Flex>
        <Flex gap={2}>
          <Button colorScheme="invokeYellow">Add Model</Button>
          <Button>Scan for Models</Button>
        </Flex>
      </Flex>
      <Box layerStyle="second" p={3} borderRadius="base" w="full" h="full">
        <Flex gap={2} alignItems="center" justifyContent="space-between">
          <Button>All Models</Button>
          <Spacer />
          <InputGroup>
            <Input placeholder={t('boards.searchBoard')} data-testid="board-search-input" />(
            <InputRightElement h="full" pe={2}>
              <IconButton size="sm" variant="link" aria-label={t('boards.clearSearch')} icon={<PiXBold />} />
            </InputRightElement>
            )
          </InputGroup>
        </Flex>
      </Box>
    </Box>
  );
};
