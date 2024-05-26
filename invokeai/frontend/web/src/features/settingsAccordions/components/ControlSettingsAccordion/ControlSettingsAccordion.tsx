import { Button, ButtonGroup, Divider, Flex, StandaloneAccordion } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import ControlAdapterConfig from 'features/controlAdapters/components/ControlAdapterConfig';
import { useAddControlAdapter } from 'features/controlAdapters/hooks/useAddControlAdapter';
import {
  selectAllControlNets,
  selectAllIPAdapters,
  selectAllT2IAdapters,
  selectControlAdapterIds,
  selectControlAdaptersSlice,
  selectValidControlNets,
  selectValidIPAdapters,
  selectValidT2IAdapters,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { useStandaloneAccordionToggle } from 'features/settingsAccordions/hooks/useStandaloneAccordionToggle';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { Fragment, memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

const selector = createMemoizedSelector([selectControlAdaptersSlice], (controlAdapters) => {
  const badges: string[] = [];
  let isError = false;

  const enabledNonRegionalIPAdapterCount = selectAllIPAdapters(controlAdapters).filter((ca) => ca.isEnabled).length;

  const validIPAdapterCount = selectValidIPAdapters(controlAdapters).length;
  if (enabledNonRegionalIPAdapterCount > 0) {
    badges.push(`${enabledNonRegionalIPAdapterCount} IP`);
  }
  if (enabledNonRegionalIPAdapterCount > validIPAdapterCount) {
    isError = true;
  }

  const enabledControlNetCount = selectAllControlNets(controlAdapters).filter((ca) => ca.isEnabled).length;
  const validControlNetCount = selectValidControlNets(controlAdapters).length;
  if (enabledControlNetCount > 0) {
    badges.push(`${enabledControlNetCount} ControlNet`);
  }
  if (enabledControlNetCount > validControlNetCount) {
    isError = true;
  }

  const enabledT2IAdapterCount = selectAllT2IAdapters(controlAdapters).filter((ca) => ca.isEnabled).length;
  const validT2IAdapterCount = selectValidT2IAdapters(controlAdapters).length;
  if (enabledT2IAdapterCount > 0) {
    badges.push(`${enabledT2IAdapterCount} T2I`);
  }
  if (enabledT2IAdapterCount > validT2IAdapterCount) {
    isError = true;
  }

  const controlAdapterIds = selectControlAdapterIds(controlAdapters);

  return {
    controlAdapterIds,
    badges,
    isError, // TODO: Add some visual indicator that the control adapters are in an error state
  };
});

export const ControlSettingsAccordion: React.FC = memo(() => {
  const { t } = useTranslation();
  const { controlAdapterIds, badges } = useAppSelector(selector);
  const isControlNetEnabled = useFeatureStatus('controlNet');
  const { isOpen, onToggle } = useStandaloneAccordionToggle({
    id: 'control-settings',
    defaultIsOpen: true,
  });
  const [addControlNet, isAddControlNetDisabled] = useAddControlAdapter('controlnet');
  const [addIPAdapter, isAddIPAdapterDisabled] = useAddControlAdapter('ip_adapter');
  const [addT2IAdapter, isAddT2IAdapterDisabled] = useAddControlAdapter('t2i_adapter');

  if (!isControlNetEnabled) {
    return null;
  }

  return (
    <StandaloneAccordion label={t('accordions.control.title')} badges={badges} isOpen={isOpen} onToggle={onToggle}>
      <Flex gap={2} p={4} flexDir="column" data-testid="control-accordion">
        <ButtonGroup size="sm" w="full" justifyContent="space-between" variant="ghost" isAttached={false}>
          <Button
            tooltip={t('controlnet.addControlNet')}
            leftIcon={<PiPlusBold />}
            onClick={addControlNet}
            data-testid="add controlnet"
            flexGrow={1}
            isDisabled={isAddControlNetDisabled}
          >
            {t('common.controlNet')}
          </Button>
          <Button
            tooltip={t('controlnet.addIPAdapter')}
            leftIcon={<PiPlusBold />}
            onClick={addIPAdapter}
            data-testid="add ip adapter"
            flexGrow={1}
            isDisabled={isAddIPAdapterDisabled}
          >
            {t('common.ipAdapter')}
          </Button>
          <Button
            tooltip={t('controlnet.addT2IAdapter')}
            leftIcon={<PiPlusBold />}
            onClick={addT2IAdapter}
            data-testid="add t2i adapter"
            flexGrow={1}
            isDisabled={isAddT2IAdapterDisabled}
          >
            {t('common.t2iAdapter')}
          </Button>
        </ButtonGroup>
        {controlAdapterIds.map((id, i) => (
          <Fragment key={id}>
            <Divider />
            <ControlAdapterConfig id={id} number={i + 1} />
          </Fragment>
        ))}
      </Flex>
    </StandaloneAccordion>
  );
});

ControlSettingsAccordion.displayName = 'ControlAdaptersSettingsAccordion';
