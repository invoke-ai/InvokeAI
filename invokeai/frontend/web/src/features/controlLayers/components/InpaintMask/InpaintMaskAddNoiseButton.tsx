import { Button, Flex } from '@invoke-ai/ui-library';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useAddInpaintMaskNoise } from 'features/controlLayers/hooks/addLayerHooks';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const InpaintMaskAddNoiseButton = () => {
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const { t } = useTranslation();
  const addInpaintMaskNoise = useAddInpaintMaskNoise(entityIdentifier);

  return (
    <Flex w="full" p={2} justifyContent="center">
      <Button size="sm" variant="ghost" leftIcon={<PiPlusBold />} onClick={addInpaintMaskNoise}>
        {t('controlLayers.imageNoise')}
      </Button>
    </Flex>
  );
};
