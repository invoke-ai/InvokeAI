import { Feature } from 'app/features';
import { useAppDispatch, useAppSelector } from 'app/storeHooks';
import FaceRestoreSettings from 'features/parameters/components/AdvancedParameters/FaceRestore/FaceRestoreSettings';
import FaceRestoreToggle from 'features/parameters/components/AdvancedParameters/FaceRestore/FaceRestoreToggle';
import ImageFit from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageFit';
import ImageToImageStrength from 'features/parameters/components/AdvancedParameters/ImageToImage/ImageToImageStrength';
import ImageToImageOutputSettings from 'features/parameters/components/AdvancedParameters/Output/ImageToImageOutputSettings';
import SeedSettings from 'features/parameters/components/AdvancedParameters/Seed/SeedSettings';
import UpscaleSettings from 'features/parameters/components/AdvancedParameters/Upscale/UpscaleSettings';
import UpscaleToggle from 'features/parameters/components/AdvancedParameters/Upscale/UpscaleToggle';
import GenerateVariationsToggle from 'features/parameters/components/AdvancedParameters/Variations/GenerateVariations';
import VariationsSettings from 'features/parameters/components/AdvancedParameters/Variations/VariationsSettings';
import MainSettings from 'features/parameters/components/MainParameters/MainParameters';
import ParametersAccordion from 'features/parameters/components/ParametersAccordion';
import ProcessButtons from 'features/parameters/components/ProcessButtons/ProcessButtons';
import PromptInput from 'features/parameters/components/PromptInput/PromptInput';
import { setHiresFix } from 'features/parameters/store/postprocessingSlice';
import InvokeOptionsPanel from 'features/ui/components/InvokeParametersPanel';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useEffect } from 'react';
import { useTranslation } from 'react-i18next';

export default function ImageToImagePanel() {
  const { t } = useTranslation();

  const imageToImageAccordions = {
    seed: {
      header: `${t('parameters:seed')}`,
      feature: Feature.SEED,
      content: <SeedSettings />,
    },
    variations: {
      header: `${t('parameters:variations')}`,
      feature: Feature.VARIATIONS,
      content: <VariationsSettings />,
      additionalHeaderComponents: <GenerateVariationsToggle />,
    },
    face_restore: {
      header: `${t('parameters:faceRestoration')}`,
      feature: Feature.FACE_CORRECTION,
      content: <FaceRestoreSettings />,
      additionalHeaderComponents: <FaceRestoreToggle />,
    },
    upscale: {
      header: `${t('parameters:upscaling')}`,
      feature: Feature.UPSCALE,
      content: <UpscaleSettings />,
      additionalHeaderComponents: <UpscaleToggle />,
    },
    other: {
      header: `${t('parameters:otherOptions')}`,
      feature: Feature.OTHER,
      content: <ImageToImageOutputSettings />,
    },
  };

  const dispatch = useAppDispatch();

  const activeTabName = useAppSelector(activeTabNameSelector);

  useEffect(() => {
    if (activeTabName === 'img2img') {
      const handleChangeHiresFix = () => dispatch(setHiresFix(false));
      handleChangeHiresFix();
    }
  }, [activeTabName, dispatch]);

  return (
    <InvokeOptionsPanel>
      <PromptInput />
      <ProcessButtons />
      <MainSettings />
      <ImageToImageStrength
        label={t('parameters:img2imgStrength')}
        styleClass="main-settings-block image-to-image-strength-main-option"
      />
      <ImageFit />
      <ParametersAccordion accordionInfo={imageToImageAccordions} />
    </InvokeOptionsPanel>
  );
}
