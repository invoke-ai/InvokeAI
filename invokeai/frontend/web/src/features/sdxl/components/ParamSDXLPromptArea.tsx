import { Flex } from '@chakra-ui/react';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import ParamSDXLConcatButton from './ParamSDXLConcatButton';
import ParamSDXLNegativeStyleConditioning from './ParamSDXLNegativeStyleConditioning';
import ParamSDXLPositiveStyleConditioning from './ParamSDXLPositiveStyleConditioning';

export default function ParamSDXLPromptArea() {
  return (
    <Flex
      sx={{
        flexDirection: 'column',
        gap: 2,
      }}
    >
      <ParamPositiveConditioning />
      <ParamSDXLConcatButton />
      <ParamSDXLPositiveStyleConditioning />
      <ParamNegativeConditioning />
      <ParamSDXLNegativeStyleConditioning />
    </Flex>
  );
}
