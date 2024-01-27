import type { FormLabelProps } from '@invoke-ai/ui-library';
import { Flex, FormControlGroup, StandaloneAccordion, Text } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ParamSDXLRefinerCFGScale from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerCFGScale';
import ParamSDXLRefinerModelSelect from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerModelSelect';
import ParamSDXLRefinerNegativeAestheticScore from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerNegativeAestheticScore';
import ParamSDXLRefinerPositiveAestheticScore from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerPositiveAestheticScore';
import ParamSDXLRefinerScheduler from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerScheduler';
import ParamSDXLRefinerStart from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerStart';
import ParamSDXLRefinerSteps from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerSteps';
import { selectSdxlSlice } from 'features/sdxl/store/sdxlSlice';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { isNil } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useIsRefinerAvailable } from 'services/api/hooks/useIsRefinerAvailable';

const aestheticLabelProps: FormLabelProps = {
  minW: '9.2rem',
};

const stepsScaleLabelProps: FormLabelProps = {
  minW: '5rem',
};

const selectBadges = createMemoizedSelector(selectSdxlSlice, (sdxl) => (sdxl.refinerModel ? ['Enabled'] : undefined));

export const RefinerSettingsAccordion: React.FC = memo(() => {
  const { t } = useTranslation();
  const isRefinerAvailable = useIsRefinerAvailable();
  const badges = useAppSelector(selectBadges);
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: 'refiner-settings',
    defaultIsOpen: false,
  });

  return (
    <StandaloneAccordion label={t('sdxl.refiner')} badges={badges} isOpen={isOpen} onToggle={onToggle}>
      {isRefinerAvailable ? <RefinerSettingsAccordionContent /> : <RefinerSettingsAccordionNoRefiner />}
    </StandaloneAccordion>
  );
});

RefinerSettingsAccordion.displayName = 'RefinerSettingsAccordion';

const RefinerSettingsAccordionNoRefiner: React.FC = memo(() => {
  const { t } = useTranslation();
  return (
    <Flex justifyContent="center" p={4}>
      <Text fontSize="sm" color="base.500">
        {t('models.noRefinerModelsInstalled')}
      </Text>
    </Flex>
  );
});

RefinerSettingsAccordionNoRefiner.displayName = 'RefinerSettingsAccordionNoRefiner';

const RefinerSettingsAccordionContent: React.FC = memo(() => {
  const isRefinerModelSelected = useAppSelector((state) => !isNil(state.sdxl.refinerModel));

  return (
    <FormControlGroup isDisabled={!isRefinerModelSelected}>
      <Flex p={4} gap={4} flexDir="column">
        <ParamSDXLRefinerModelSelect />
        <FormControlGroup formLabelProps={stepsScaleLabelProps} isDisabled={!isRefinerModelSelected}>
          <ParamSDXLRefinerScheduler />
          <ParamSDXLRefinerSteps />
          <ParamSDXLRefinerCFGScale />
          <ParamSDXLRefinerStart />
        </FormControlGroup>
        <FormControlGroup formLabelProps={aestheticLabelProps} isDisabled={!isRefinerModelSelected}>
          <ParamSDXLRefinerPositiveAestheticScore />
          <ParamSDXLRefinerNegativeAestheticScore />
        </FormControlGroup>
      </Flex>
    </FormControlGroup>
  );
});

RefinerSettingsAccordionContent.displayName = 'RefinerSettingsAccordionContent';
