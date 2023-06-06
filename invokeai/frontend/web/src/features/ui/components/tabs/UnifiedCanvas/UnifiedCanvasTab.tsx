import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import UnifiedCanvasContent from './UnifiedCanvasContent';
import UnifiedCanvasParameters from './UnifiedCanvasParameters';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';

const UnifiedCanvasTab = () => {
  return (
    <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
      <ParametersPinnedWrapper>
        <UnifiedCanvasParameters />
      </ParametersPinnedWrapper>
      <UnifiedCanvasContent />
    </Flex>
  );
};

export default memo(UnifiedCanvasTab);
