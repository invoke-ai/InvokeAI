import { Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import ParamSpandrelModel from '../../../parameters/components/Upscale/ParamSpandrelModel';
import { UpscaleInitialImage } from './UpscaleInitialImage';

export const UpscaleSettingsAccordion = memo(() => {
  const { t } = useTranslation();
  const { isOpen: isOpenAccordion, onToggle: onToggleAccordion } = useStandaloneAccordionToggle({
    id: 'upscale-settings',
    defaultIsOpen: true,
  });

  return (
    <StandaloneAccordion label="Upscale" isOpen={isOpenAccordion} onToggle={onToggleAccordion}>
      <Flex p={4} w="full" h="full" flexDir="column" data-testid="image-settings-accordion">
        <Flex gap={4}>
          <UpscaleInitialImage />
          <ParamSpandrelModel />
        </Flex>
      </Flex>
    </StandaloneAccordion>
  );
});

UpscaleSettingsAccordion.displayName = 'UpscaleSettingsAccordion';
