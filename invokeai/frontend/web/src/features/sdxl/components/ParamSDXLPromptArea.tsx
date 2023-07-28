import { Box, Flex } from '@chakra-ui/react';
import { RootState } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import ParamSDXLConcatButton from './ParamSDXLConcatButton';
import ParamSDXLNegativeStyleConditioning from './ParamSDXLNegativeStyleConditioning';
import ParamSDXLPositiveStyleConditioning from './ParamSDXLPositiveStyleConditioning';
import SDXLConcatLink from './SDXLConcatLink';

export default function ParamSDXLPromptArea() {
  const shouldPinParametersPanel = useAppSelector(
    (state: RootState) => state.ui.shouldPinParametersPanel
  );

  return (
    <Flex
      sx={{
        flexDirection: 'column',
        gap: 2,
      }}
    >
      <Box
        sx={{
          position: 'absolute',
          w: 'full',
          top: shouldPinParametersPanel ? '131px' : '187px',
        }}
      >
        <SDXLConcatLink />
      </Box>
      <Box
        sx={{
          position: 'absolute',
          w: 'full',
          top: shouldPinParametersPanel ? '275px' : '331px',
        }}
      >
        <SDXLConcatLink />
      </Box>
      <ParamPositiveConditioning />
      <ParamSDXLConcatButton />
      <ParamSDXLPositiveStyleConditioning />
      <ParamNegativeConditioning />
      <ParamSDXLNegativeStyleConditioning />
    </Flex>
  );
}
