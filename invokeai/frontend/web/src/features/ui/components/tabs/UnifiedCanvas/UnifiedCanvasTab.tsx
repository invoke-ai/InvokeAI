import { Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import SDXLUnifiedCanvasTabParameters from 'features/sdxl/components/SDXLUnifiedCanvasTabParameters';
import { memo } from 'react';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';
import UnifiedCanvasContent from './UnifiedCanvasContent';
import UnifiedCanvasParameters from './UnifiedCanvasParameters';

const UnifiedCanvasTab = () => {
  const model = useAppSelector((state: RootState) => state.generation.model);
  return (
    <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
      <ParametersPinnedWrapper>
        {model && model.base_model === 'sdxl' ? (
          <SDXLUnifiedCanvasTabParameters />
        ) : (
          <UnifiedCanvasParameters />
        )}
      </ParametersPinnedWrapper>
      <UnifiedCanvasContent />
    </Flex>
  );
};

export default memo(UnifiedCanvasTab);
