// import { Button, Flex } from '@invoke-ai/ui-library';
// import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
// import { useAddInpaintMaskDenoiseLimit, useAddInpaintMaskNoise } from 'features/controlLayers/hooks/addLayerHooks';
// import { useTranslation } from 'react-i18next';
// import { PiPlusBold } from 'react-icons/pi';

// Removed buttons because denosie limit is not helpful for many architectures
// Users can access with right click menu instead.
// If buttons for noise or new features are deemed important in the future, add them back here.
export const InpaintMaskAddButtons = () => {
  // Buttons are temporarily hidden. To restore, uncomment the code below.
  return null;
  // const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  // const { t } = useTranslation();
  // const addInpaintMaskDenoiseLimit = useAddInpaintMaskDenoiseLimit(entityIdentifier);
  // const addInpaintMaskNoise = useAddInpaintMaskNoise(entityIdentifier);
  // return (
  //   <Flex w="full" p={2} justifyContent="center">
  //     <Button size="sm" variant="ghost" leftIcon={<PiPlusBold />} onClick={addInpaintMaskDenoiseLimit}>
  //       {t('controlLayers.denoiseLimit')}
  //     </Button>
  //     <Button size="sm" variant="ghost" leftIcon={<PiPlusBold />} onClick={addInpaintMaskNoise}>
  //       {t('controlLayers.imageNoise')}
  //     </Button>
  //   </Flex>
  // );
};
