import { Flex } from '@chakra-ui/react';
import ParamDynamicPromptsCollapse from 'features/dynamicPrompts/components/ParamDynamicPromptsCollapse';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import ParamNoiseCollapse from 'features/parameters/components/Parameters/Noise/ParamNoiseCollapse';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import TextToImageTabCoreParameters from 'features/ui/components/tabs/TextToImage/TextToImageTabCoreParameters';
import ParamSDXLConcatPrompt from './ParamSDXLConcatPrompt';
import ParamSDXLNegativeStyleConditioning from './ParamSDXLNegativeStyleConditioning';
import ParamSDXLPositiveStyleConditioning from './ParamSDXLPositiveStyleConditioning';
import ParamSDXLRefinerCollapse from './ParamSDXLRefinerCollapse';

const SDXLTextToImageTabParameters = () => {
  return (
    <>
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
        <ParamSDXLConcatPrompt />
      </Flex>
      <ProcessButtons />
      <TextToImageTabCoreParameters />
      <ParamSDXLRefinerCollapse />
      <ParamDynamicPromptsCollapse />
      <ParamNoiseCollapse />
    </>
  );
};

export default SDXLTextToImageTabParameters;
