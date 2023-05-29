import { Flex } from '@chakra-ui/react';
import IAIScroller from 'common/components/IAIScroller';
import ControlNet from 'features/controlnet/ControlNet';
import { memo } from 'react';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';
import TextToImageTabMain from './TextToImageTabMain';
import TextToImageTabParameters from './TextToImageTabParameters';

const TextToImageTab = () => {
  return (
    <>
      <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
        <ParametersPinnedWrapper>
          <TextToImageTabParameters />
        </ParametersPinnedWrapper>
        <IAIScroller width="30%">
          <ControlNet />
        </IAIScroller>
        <TextToImageTabMain />
      </Flex>
    </>
  );
};

export default memo(TextToImageTab);
