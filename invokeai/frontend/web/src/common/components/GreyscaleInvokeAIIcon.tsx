import { Box, Image } from '@chakra-ui/react';
import InvokeAILogoImage from 'assets/images/logo.png';
import { memo } from 'react';

const GreyscaleInvokeAIIcon = () => (
  <Box pos="relative" w={4} h={4}>
    <Image
      src={InvokeAILogoImage}
      alt="invoke-ai-logo"
      pos="absolute"
      top={-0.5}
      insetInlineStart={-0.5}
      w={5}
      h={5}
      minW={5}
      minH={5}
      filter="saturate(0)"
    />
  </Box>
);

export default memo(GreyscaleInvokeAIIcon);
