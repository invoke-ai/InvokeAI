import { Flex, UnorderedList, ListItem, Icon, Text } from '@invoke-ai/ui-library';
import { PiArrowSquareOutBold } from 'react-icons/pi';

export const CanvasV2Announcement = () => {
  return (
    <Flex gap={4} flexDir="column">
      <UnorderedList fontSize="sm">
        <ListItem>A poweful new control canvas</ListItem>
        <ListItem>New layer types for even more control</ListItem>
        <ListItem>Support for the Flux family of models</ListItem>
      </UnorderedList>
      <Flex flexDir="column" gap={1}>
        <Flex gap={2}>
          <Text as="a" target="_blank" href="" fontWeight="semibold">
            Read Release Notes
          </Text>
          <Icon as={PiArrowSquareOutBold} />
        </Flex>
        <Flex gap={2}>
          <Text as="a" target="_blank" href="" fontWeight="semibold">
            Watch Release Video
          </Text>
          <Icon as={PiArrowSquareOutBold} />
        </Flex>
        <Flex gap={2}>
          <Text as="a" target="_blank" href="" fontWeight="semibold">
            Watch UI Updates Overview
          </Text>
          <Icon as={PiArrowSquareOutBold} />
        </Flex>
      </Flex>
    </Flex>
  );
};
