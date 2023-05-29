import { Flex } from '@chakra-ui/react';
import ControlNet from 'features/controlnet/ControlNet';
import { memo } from 'react';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';
import TextToImageTabMain from './TextToImageTabMain';
import TextToImageTabParameters from './TextToImageTabParameters';

const TextToImageTab = () => {
  return (
    <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
      <ParametersPinnedWrapper>
        <TextToImageTabParameters />
      </ParametersPinnedWrapper>
      <ControlNet />
      <TextToImageTabMain />
    </Flex>
  );
};

export default memo(TextToImageTab);
