import { Box, Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import { AnimatePresence } from 'framer-motion';
import ParamSDXLConcatButton from './ParamSDXLConcatButton';
import ParamSDXLNegativeStyleConditioning from './ParamSDXLNegativeStyleConditioning';
import ParamSDXLPositiveStyleConditioning from './ParamSDXLPositiveStyleConditioning';
import SDXLConcatLink from './SDXLConcatLink';

export default function ParamSDXLPromptArea() {
  const shouldPinParametersPanel = useAppSelector(
    (state: RootState) => state.ui.shouldPinParametersPanel
  );

  const shouldConcatSDXLStylePrompt = useAppSelector(
    (state: RootState) => state.sdxl.shouldConcatSDXLStylePrompt
  );

  return (
    <Flex
      sx={{
        flexDirection: 'column',
        gap: 2,
      }}
    >
      <AnimatePresence>
        {shouldConcatSDXLStylePrompt && (
          <Box
            sx={{
              position: 'absolute',
              w: 'full',
              top: shouldPinParametersPanel ? '119px' : '175px',
            }}
          >
            <SDXLConcatLink />
          </Box>
        )}
      </AnimatePresence>
      <AnimatePresence>
        {shouldConcatSDXLStylePrompt && (
          <Box
            sx={{
              position: 'absolute',
              w: 'full',
              top: shouldPinParametersPanel ? '263px' : '319px',
            }}
          >
            <SDXLConcatLink />
          </Box>
        )}
      </AnimatePresence>
      <ParamPositiveConditioning />
      <ParamSDXLConcatButton />
      <ParamSDXLPositiveStyleConditioning />
      <ParamNegativeConditioning />
      <ParamSDXLNegativeStyleConditioning />
    </Flex>
  );
}
