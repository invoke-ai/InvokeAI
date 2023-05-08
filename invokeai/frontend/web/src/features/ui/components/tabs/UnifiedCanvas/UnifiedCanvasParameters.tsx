import { Flex } from '@chakra-ui/react';
import { Feature } from 'app/features';
import BoundingBoxSettings from 'features/parameters/components/AdvancedParameters/Canvas/BoundingBox/BoundingBoxSettings';
import InfillAndScalingSettings from 'features/parameters/components/AdvancedParameters/Canvas/InfillAndScalingSettings';
import SeamCorrectionSettings from 'features/parameters/components/AdvancedParameters/Canvas/SeamCorrection/SeamCorrectionSettings';
import ImageToImageStrength from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageStrength';
import MainSettings from 'features/parameters/components/MainParameters/MainSettings';
import ParametersAccordion, {
  ParametersAccordionItems,
} from 'features/parameters/components/ParametersAccordion';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import { useTranslation } from 'react-i18next';
import OverlayScrollable from '../../common/OverlayScrollable';
// import ParamSeedSettings from 'features/parameters/components/Parameters/Seed/ParamSeedSettings';
// import ParamVariationSettings from 'features/parameters/components/Parameters/Variations/ParamVariationSettings';
// import ParamSymmetrySettings from 'features/parameters/components/Parameters/Symmetry/ParamSymmetrySettings';
// import ParamVariationToggle from 'features/parameters/components/Parameters/Variations/ParamVariationToggle';
// import ParamSymmetryToggle from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryToggle';
import ParamPositiveConditioning from 'features/parameters/components/Parameters/ParamPositiveConditioning';
import ParamNegativeConditioning from 'features/parameters/components/Parameters/ParamNegativeConditioning';
import ParamSeedCollapse from 'features/parameters/components/Parameters/Seed/ParamSeedCollapse';
import ParamVariationCollapse from 'features/parameters/components/Parameters/Variations/ParamVariationCollapse';
import ParamSymmetryCollapse from 'features/parameters/components/Parameters/Symmetry/ParamSymmetryCollapse';

export default function UnifiedCanvasParameters() {
  const { t } = useTranslation();

  const unifiedCanvasAccordions: ParametersAccordionItems = {
    general: {
      name: 'general',
      header: `${t('parameters.general')}`,
      feature: undefined,
      content: <MainSettings />,
    },
    unifiedCanvasImg2Img: {
      name: 'unifiedCanvasImg2Img',
      header: `${t('parameters.imageToImage')}`,
      feature: undefined,
      content: <ImageToImageStrength />,
    },
    // seed: {
    //   name: 'seed',
    //   header: `${t('parameters.seed')}`,
    //   feature: Feature.SEED,
    //   content: <ParamSeedSettings />,
    // },
    boundingBox: {
      name: 'boundingBox',
      header: `${t('parameters.boundingBoxHeader')}`,
      feature: Feature.BOUNDING_BOX,
      content: <BoundingBoxSettings />,
    },
    seamCorrection: {
      name: 'seamCorrection',
      header: `${t('parameters.seamCorrectionHeader')}`,
      feature: Feature.SEAM_CORRECTION,
      content: <SeamCorrectionSettings />,
    },
    infillAndScaling: {
      name: 'infillAndScaling',
      header: `${t('parameters.infillScalingHeader')}`,
      feature: Feature.INFILL_AND_SCALING,
      content: <InfillAndScalingSettings />,
    },
    // variations: {
    //   name: 'variations',
    //   header: `${t('parameters.variations')}`,
    //   feature: Feature.VARIATIONS,
    //   content: <ParamVariationSettings />,
    //   additionalHeaderComponents: <ParamVariationToggle />,
    // },
    // symmetry: {
    //   name: 'symmetry',
    //   header: `${t('parameters.symmetry')}`,
    //   content: <ParamSymmetrySettings />,
    //   additionalHeaderComponents: <ParamSymmetryToggle />,
    // },
  };

  return (
    <OverlayScrollable>
      <Flex
        sx={{
          gap: 2,
          flexDirection: 'column',
          h: 'full',
          w: 'full',
          position: 'absolute',
        }}
      >
        <ParamPositiveConditioning />
        <ParamNegativeConditioning />
        <ProcessButtons />
        <ParamSeedCollapse />
        <ParamVariationCollapse />
        <ParamSymmetryCollapse />
        <ParametersAccordion accordionItems={unifiedCanvasAccordions} />
      </Flex>
    </OverlayScrollable>
  );
}
