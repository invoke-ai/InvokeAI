import { Box, Flex } from '@chakra-ui/react';

export default function ControlNetImage() {
  return (
    <Flex gap={2}>
      <Box width="50%" height={200} bg="base.850">
        Image
      </Box>
      <Box width="50%" height={200} bg="base.850">
        Preview
      </Box>
    </Flex>
  );
}
