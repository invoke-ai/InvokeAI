import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';
import TextToImageTabMain from './TextToImageTabMain';
import TextToImageTabParameters from './TextToImageTabParameters';

const TextToImageTab = () => {
  return (
    <Flex sx={{ gap: 2, w: 'full', h: 'full' }}>
      <ParametersPinnedWrapper>
        <TextToImageTabParameters />
      </ParametersPinnedWrapper>
      <TextToImageTabMain />
    </Flex>
  );
};

export default memo(TextToImageTab);
