import { Button, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  buildSelectValidRegionalGuidanceActions,
  useAddRegionalGuidanceIPAdapter,
  useAddRegionalGuidanceNegativePrompt,
  useAddRegionalGuidancePositivePrompt,
} from 'features/controlLayers/hooks/addLayerHooks';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const RegionalGuidanceAddPromptsIPAdapterButtons = () => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const { t } = useTranslation();
  const addRegionalGuidanceIPAdapter = useAddRegionalGuidanceIPAdapter(entityIdentifier);
  const addRegionalGuidancePositivePrompt = useAddRegionalGuidancePositivePrompt(entityIdentifier);
  const addRegionalGuidanceNegativePrompt = useAddRegionalGuidanceNegativePrompt(entityIdentifier);

  const selectValidActions = useMemo(
    () => buildSelectValidRegionalGuidanceActions(entityIdentifier),
    [entityIdentifier]
  );
  const validActions = useAppSelector(selectValidActions);

  return (
    <Flex w="full" p={2} justifyContent="space-between">
      <Button
        size="sm"
        variant="ghost"
        leftIcon={<PiPlusBold />}
        onPointerUp={addRegionalGuidancePositivePrompt}
        isDisabled={!validActions.canAddPositivePrompt}
      >
        {t('controlLayers.prompt')}
      </Button>
      <Button
        size="sm"
        variant="ghost"
        leftIcon={<PiPlusBold />}
        onPointerUp={addRegionalGuidanceNegativePrompt}
        isDisabled={!validActions.canAddNegativePrompt}
      >
        {t('controlLayers.negativePrompt')}
      </Button>
      <Button size="sm" variant="ghost" leftIcon={<PiPlusBold />} onPointerUp={addRegionalGuidanceIPAdapter}>
        {t('controlLayers.referenceImage')}
      </Button>
    </Flex>
  );
};
