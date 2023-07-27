import { Flex } from '@chakra-ui/react';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import ParamSDXLConcatPrompt from './ParamSDXLConcatPrompt';
import ParamSDXLNegativeStyleConditioning from './ParamSDXLNegativeStyleConditioning';
import ParamSDXLPositiveStyleConditioning from './ParamSDXLPositiveStyleConditioning';
import ParamSDXLStylePresetSelect from './ParamSDXLStylePresetSelect';

export default function ParamSDXLPromptArea() {
  return (
    <Flex
      sx={{
        flexDirection: 'column',
        gap: 2,
        p: 2,
        borderRadius: 4,
        bg: 'base.100',
        _dark: { bg: 'base.850' },
      }}
    >
      <ParamPositiveConditioning />
      <ParamSDXLPositiveStyleConditioning />
      <ParamNegativeConditioning />
      <ParamSDXLNegativeStyleConditioning />
      <ParamSDXLStylePresetSelect />
      <ParamSDXLConcatPrompt />
    </Flex>
  );
}
