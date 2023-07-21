import { Badge, Flex } from '@chakra-ui/react';

const AutoAddIcon = () => {
  return (
    <Flex
      sx={{
        position: 'absolute',
        insetInlineStart: 0,
        top: 0,
        p: 1,
      }}
    >
      <Badge
        variant="solid"
        sx={{ fontSize: 10, bg: 'accent.400', _dark: { bg: 'accent.500' } }}
      >
        auto
      </Badge>
    </Flex>
  );
};

export default AutoAddIcon;
