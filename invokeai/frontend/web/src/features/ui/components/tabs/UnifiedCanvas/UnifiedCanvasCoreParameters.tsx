import { Box, Flex } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import IAICollapse from 'common/components/IAICollapse';
import ParamBoundingBoxSize from 'features/parameters/components/Parameters/Canvas/BoundingBox/ParamBoundingBoxSize';
import ParamCFGScale from 'features/parameters/components/Parameters/Core/ParamCFGScale';
import ParamIterations from 'features/parameters/components/Parameters/Core/ParamIterations';
import ParamModelandVAEandScheduler from 'features/parameters/components/Parameters/Core/ParamModelandVAEandScheduler';
import ParamSteps from 'features/parameters/components/Parameters/Core/ParamSteps';
import ImageToImageStrength from 'features/parameters/components/Parameters/ImageToImage/ImageToImageStrength';
import ParamSeedFull from 'features/parameters/components/Parameters/Seed/ParamSeedFull';
import { useCoreParametersCollapseLabel } from 'features/parameters/util/useCoreParametersCollapseLabel';
import { memo } from 'react';

const UnifiedCanvasCoreParameters = () => {
  const shouldUseSliders = useAppSelector((state) => state.ui.shouldUseSliders);
  const { iterationsAndSeedLabel } = useCoreParametersCollapseLabel();

  return (
    <IAICollapse
      label="General"
      activeLabel={iterationsAndSeedLabel}
      defaultIsOpen={true}
    >
      <Flex
        sx={{
          flexDirection: 'column',
          gap: 3,
        }}
      >
        {shouldUseSliders ? (
          <>
            <ParamIterations />
            <ParamSteps />
            <ParamCFGScale />
            <ParamModelandVAEandScheduler />
            <Box pt={2}>
              <ParamSeedFull />
            </Box>
            <ParamBoundingBoxSize />
          </>
        ) : (
          <>
            <Flex gap={3}>
              <ParamIterations />
              <ParamSteps />
              <ParamCFGScale />
            </Flex>
            <ParamModelandVAEandScheduler />
            <Box pt={2}>
              <ParamSeedFull />
            </Box>
            <ParamBoundingBoxSize />
          </>
        )}
        <ImageToImageStrength />
      </Flex>
    </IAICollapse>
  );
};

export default memo(UnifiedCanvasCoreParameters);
