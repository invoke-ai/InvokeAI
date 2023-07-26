import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import TextToImageSDXLTabParameters from 'features/sdxl/components/SDXLTextToImageTabParameters';
import { memo } from 'react';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';
import TextToImageTabMain from './TextToImageTabMain';
import TextToImageTabParameters from './TextToImageTabParameters';

const TextToImageTab = () => {
  const model = useAppSelector((state: RootState) => state.generation.model);
  return (
    <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
      <ParametersPinnedWrapper>
        {model && model.base_model === 'sdxl' ? (
          <TextToImageSDXLTabParameters />
        ) : (
          <TextToImageTabParameters />
        )}
      </ParametersPinnedWrapper>
      <TextToImageTabMain />
    </Flex>
  );
};

export default memo(TextToImageTab);
