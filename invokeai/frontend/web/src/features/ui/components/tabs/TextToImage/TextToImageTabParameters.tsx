import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import { memo } from 'react';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/Core/ParamPositiveConditioning';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/Core/ParamNegativeConditioning';
import ParamSeedCollapse from 'features/parameters/components/Parameters/Seed/ParamSeedCollapse';
import ParamVariationCollapse from 'features/parameters/components/Parameters/Variations/ParamVariationCollapse';
import ParamNoiseCollapse from 'features/parameters/components/Parameters/Noise/ParamNoiseCollapse';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';
import ParamHiresCollapse from 'features/parameters/components/Parameters/Hires/ParamHiresCollapse';
import ParamSeamlessCollapse from 'features/parameters/components/Parameters/Seamless/ParamSeamlessCollapse';
import TextToImageTabCoreParameters from './TextToImageTabCoreParameters';

const TextToImageTabParameters = () => {
  return (
    <>
      <ParamPositiveConditioning />
      <ParamNegativeConditioning />
      <ProcessButtons />
      <TextToImageTabCoreParameters />
      <ParamSeedCollapse />
      <ParamVariationCollapse />
      <ParamNoiseCollapse />
      <ParamSymmetryCollapse />
      <ParamHiresCollapse />
      <ParamSeamlessCollapse />
    </>
  );
};

export default memo(TextToImageTabParameters);
