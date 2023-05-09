import { Flex } from '@chakra-ui/react';
import { memo } from 'react';
import { createSelector } from '@reduxjs/toolkit';
import { uiSelector } from 'features/ui/store/uiSelectors';
import { useAppSelector } from 'app/store/storeHooks';
import UnifiedCanvasContent from './UnifiedCanvasContent';
import UnifiedCanvasParameters from './UnifiedCanvasParameters';
import UnifiedCanvasContentBeta from './UnifiedCanvasBeta/UnifiedCanvasContentBeta';
import ParametersPinnedWrapper from '../../ParametersPinnedWrapper';

const selector = createSelector(uiSelector, (ui) => {
  const { shouldUseCanvasBetaLayout } = ui;

  return {
    shouldUseCanvasBetaLayout,
  };
});

const UnifiedCanvasTab = () => {
  const { shouldUseCanvasBetaLayout } = useAppSelector(selector);

  return (
    <Flex sx={{ gap: 4, w: 'full', h: 'full' }}>
      <ParametersPinnedWrapper>
        <UnifiedCanvasParameters />
      </ParametersPinnedWrapper>
      {shouldUseCanvasBetaLayout ? (
        <UnifiedCanvasContentBeta />
      ) : (
        <UnifiedCanvasContent />
      )}
    </Flex>
  );
};

export default memo(UnifiedCanvasTab);
