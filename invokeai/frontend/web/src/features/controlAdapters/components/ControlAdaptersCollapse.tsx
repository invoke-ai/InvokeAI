import { ButtonGroup, Divider, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton from 'common/components/IAIButton';
import IAICollapse from 'common/components/IAICollapse';
import ControlAdapterConfig from 'features/controlAdapters/components/ControlAdapterConfig';
import {
  selectAllControlNets,
  selectAllIPAdapters,
  selectAllT2IAdapters,
  selectControlAdapterIds,
  selectValidControlNets,
  selectValidIPAdapters,
  selectValidT2IAdapters,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { Fragment, memo } from 'react';
import { useTranslation } from 'react-i18next';
import { FaPlus } from 'react-icons/fa';
import { useAddControlAdapter } from '../hooks/useAddControlAdapter';

const selector = createSelector(
  [stateSelector],
  ({ controlAdapters }) => {
    const activeLabel: string[] = [];
    let isError = false;

    const enabledIPAdapterCount = selectAllIPAdapters(controlAdapters).filter(
      (ca) => ca.isEnabled
    ).length;
    const validIPAdapterCount = selectValidIPAdapters(controlAdapters).length;
    if (enabledIPAdapterCount > 0) {
      activeLabel.push(`${enabledIPAdapterCount} IP`);
    }
    if (enabledIPAdapterCount > validIPAdapterCount) {
      isError = true;
    }

    const enabledControlNetCount = selectAllControlNets(controlAdapters).filter(
      (ca) => ca.isEnabled
    ).length;
    const validControlNetCount = selectValidControlNets(controlAdapters).length;
    if (enabledControlNetCount > 0) {
      activeLabel.push(`${enabledControlNetCount} ControlNet`);
    }
    if (enabledControlNetCount > validControlNetCount) {
      isError = true;
    }

    const enabledT2IAdapterCount = selectAllT2IAdapters(controlAdapters).filter(
      (ca) => ca.isEnabled
    ).length;
    const validT2IAdapterCount = selectValidT2IAdapters(controlAdapters).length;
    if (enabledT2IAdapterCount > 0) {
      activeLabel.push(`${enabledT2IAdapterCount} T2I`);
    }
    if (enabledT2IAdapterCount > validT2IAdapterCount) {
      isError = true;
    }

    const controlAdapterIds =
      selectControlAdapterIds(controlAdapters).map(String);

    return {
      controlAdapterIds,
      activeLabel: activeLabel.join(', '),
      isError, // TODO: Add some visual indicator that the control adapters are in an error state
    };
  },
  defaultSelectorOptions
);

const ControlAdaptersCollapse = () => {
  const { t } = useTranslation();
  const { controlAdapterIds, activeLabel } = useAppSelector(selector);
  const isControlNetDisabled = useFeatureStatus('controlNet').isFeatureDisabled;

  const [addControlNet, isAddControlNetDisabled] =
    useAddControlAdapter('controlnet');
  const [addIPAdapter, isAddIPAdapterDisabled] =
    useAddControlAdapter('ip_adapter');
  const [addT2IAdapter, isAddT2IAdapterDisabled] =
    useAddControlAdapter('t2i_adapter');

  if (isControlNetDisabled) {
    return null;
  }

  return (
    <IAICollapse
      label={t('controlnet.controlAdapter_other')}
      activeLabel={activeLabel}
    >
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ButtonGroup size="sm" w="full" justifyContent="space-between">
          <IAIButton
            tooltip={t('controlnet.addControlNet')}
            leftIcon={<FaPlus />}
            onClick={addControlNet}
            data-testid="add controlnet"
            flexGrow={1}
            isDisabled={isAddControlNetDisabled}
          >
            {t('common.controlNet')}
          </IAIButton>
          <IAIButton
            tooltip={t('controlnet.addIPAdapter')}
            leftIcon={<FaPlus />}
            onClick={addIPAdapter}
            data-testid="add ip adapter"
            flexGrow={1}
            isDisabled={isAddIPAdapterDisabled}
          >
            {t('common.ipAdapter')}
          </IAIButton>
          <IAIButton
            tooltip={t('controlnet.addT2IAdapter')}
            leftIcon={<FaPlus />}
            onClick={addT2IAdapter}
            data-testid="add t2i adapter"
            flexGrow={1}
            isDisabled={isAddT2IAdapterDisabled}
          >
            {t('common.t2iAdapter')}
          </IAIButton>
        </ButtonGroup>
        {controlAdapterIds.map((id, i) => (
          <Fragment key={id}>
            <Divider />
            <ControlAdapterConfig id={id} number={i + 1} />
          </Fragment>
        ))}
      </Flex>
    </IAICollapse>
  );
};

export default memo(ControlAdaptersCollapse);
