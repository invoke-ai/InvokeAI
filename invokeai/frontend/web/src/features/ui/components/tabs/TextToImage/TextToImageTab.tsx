import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import TextToImageTabMain from './TextToImageTabMain';
import TextToImageTabParameters from './TextToImageTabParameters';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';

const TextToImageTab = () => {
  return (
    <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
      <ParametersPinnedWrapper>
        <TextToImageTabParameters />
      </ParametersPinnedWrapper>
      <TextToImageTabMain />
    </Flex>
  );
};

export default memo(TextToImageTab);
