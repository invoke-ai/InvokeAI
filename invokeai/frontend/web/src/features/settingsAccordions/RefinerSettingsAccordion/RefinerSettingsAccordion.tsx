import { Flex } from '@chakra-ui/react';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { InvControlGroup } from 'common/components/InvControl/InvControlGroup';
import type { InvLabelProps } from 'common/components/InvControl/types';
import { InvSingleAccordion } from 'common/components/InvSingleAccordion/InvSingleAccordion';
import { InvText } from 'common/components/InvText/wrapper';
import ParamSDXLRefinerCFGScale from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerCFGScale';
import ParamSDXLRefinerModelSelect from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerModelSelect';
import ParamSDXLRefinerNegativeAestheticScore from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerNegativeAestheticScore';
import ParamSDXLRefinerPositiveAestheticScore from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerPositiveAestheticScore';
import ParamSDXLRefinerScheduler from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerScheduler';
import ParamSDXLRefinerStart from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerStart';
import ParamSDXLRefinerSteps from 'features/sdxl/components/SDXLRefiner/ParamSDXLRefinerSteps';
import { selectSdxlSlice } from 'features/sdxl/store/sdxlSlice';
import { isNil } from 'lodash-es';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { useIsRefinerAvailable } from 'services/api/hooks/useIsRefinerAvailable';

const aestheticLabelProps: InvLabelProps = {
  minW: '9.2rem',
};

const stepsScaleLabelProps: InvLabelProps = {
  minW: '5rem',
};

const selectBadges = createMemoizedSelector(selectSdxlSlice, (sdxl) =>
  sdxl.refinerModel ? ['Enabled'] : undefined
);

export const RefinerSettingsAccordion: React.FC = memo(() => {
  const { t } = useTranslation();
  const isRefinerAvailable = useIsRefinerAvailable();
  const badges = useAppSelector(selectBadges);

  if (!isRefinerAvailable) {
    return (
      <InvSingleAccordion label={t('sdxl.refiner')} badges={badges}>
        <RefinerSettingsAccordionNoRefiner />
      </InvSingleAccordion>
    );
  }

  return (
    <InvSingleAccordion label={t('sdxl.refiner')} badges={badges}>
      <RefinerSettingsAccordionContent />
    </InvSingleAccordion>
  );
});

RefinerSettingsAccordion.displayName = 'RefinerSettingsAccordion';

const RefinerSettingsAccordionNoRefiner: React.FC = memo(() => {
  const { t } = useTranslation();
  return (
    <Flex justifyContent="center" p={4}>
      <InvText fontSize="sm" color="base.500">
        {t('models.noRefinerModelsInstalled')}
      </InvText>
    </Flex>
  );
});

RefinerSettingsAccordionNoRefiner.displayName =
  'RefinerSettingsAccordionNoRefiner';

const RefinerSettingsAccordionContent: React.FC = memo(() => {
  const isRefinerModelSelected = useAppSelector(
    (state) => !isNil(state.sdxl.refinerModel)
  );

  return (
    <InvControlGroup isDisabled={!isRefinerModelSelected}>
      <Flex p={4} gap={4} flexDir="column">
        <ParamSDXLRefinerModelSelect />
        <InvControlGroup
          labelProps={stepsScaleLabelProps}
          isDisabled={!isRefinerModelSelected}
        >
          <ParamSDXLRefinerScheduler />
          <ParamSDXLRefinerSteps />
          <ParamSDXLRefinerCFGScale />
          <ParamSDXLRefinerStart />
        </InvControlGroup>
        <InvControlGroup
          labelProps={aestheticLabelProps}
          isDisabled={!isRefinerModelSelected}
        >
          <ParamSDXLRefinerPositiveAestheticScore />
          <ParamSDXLRefinerNegativeAestheticScore />
        </InvControlGroup>
      </Flex>
    </InvControlGroup>
  );
});

RefinerSettingsAccordionContent.displayName = 'RefinerSettingsAccordionContent';
