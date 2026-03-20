import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Flex, FormControlGroup, StandaloneAccordion } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { selectExternalProviderId, selectIsExternal } from 'features/controlLayers/store/paramsSlice';
import { ExternalModelImageSizeSelect } from 'features/parameters/components/Dimensions/ExternalModelImageSizeSelect';
import { ExternalModelResolutionSelect } from 'features/parameters/components/Dimensions/ExternalModelResolutionSelect';
import { GeminiProviderOptions } from 'features/parameters/components/External/GeminiProviderOptions';
import { OpenAIProviderOptions } from 'features/parameters/components/External/OpenAIProviderOptions';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';

const formLabelProps: FormLabelProps = {
  minW: '4rem',
};

export const ExternalSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const isExternal = useAppSelector(selectIsExternal);
  const providerId = useAppSelector(selectExternalProviderId);
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: 'external-settings',
    defaultIsOpen: true,
  });

  if (!isExternal) {
    return null;
  }

  return (
    <StandaloneAccordion
      label={t('accordions.advanced.title')}
      badges={['EXTERNAL']}
      isOpen={isOpen}
      onToggle={onToggle}
    >
      <Flex gap={4} p={4} flexDir="column" data-testid="external-settings-accordion">
        <FormControlGroup formLabelProps={formLabelProps}>
          <ExternalModelResolutionSelect />
          <ExternalModelImageSizeSelect />
          {providerId === 'openai' && <OpenAIProviderOptions />}
          {providerId === 'gemini' && <GeminiProviderOptions />}
        </FormControlGroup>
      </Flex>
    </StandaloneAccordion>
  );
});

ExternalSettingsAccordion.displayName = 'ExternalSettingsAccordion';
