import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';
import UnifiedCanvasContent from './UnifiedCanvasContent';
import UnifiedCanvasParameters from './UnifiedCanvasParameters';

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
