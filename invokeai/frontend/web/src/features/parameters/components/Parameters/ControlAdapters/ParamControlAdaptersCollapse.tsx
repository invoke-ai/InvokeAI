import { ButtonGroup, Divider, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton from 'common/components/IAIButton';
import IAICollapse from 'common/components/IAICollapse';
import ControlAdapterConfig from 'features/controlAdapters/components/ControlAdapterConfig';
import { useAddControlNet } from 'features/controlAdapters/hooks/useAddControlNet';
import { useAddIPAdapter } from 'features/controlAdapters/hooks/useAddIPAdapter';
import { useAddT2IAdapter } from 'features/controlAdapters/hooks/useAddT2IAdapter';
import {
  selectAllControlNets,
  selectAllIPAdapters,
  selectAllT2IAdapters,
  selectControlAdapterIds,
} from 'features/controlAdapters/store/controlAdaptersSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { Fragment, memo } from 'react';
import { FaPlus } from 'react-icons/fa';

const selector = createSelector(
  [stateSelector],
  ({ controlAdapters }) => {
    const activeLabel: string[] = [];

    const ipAdapterCount = selectAllIPAdapters(controlAdapters).length;
    if (ipAdapterCount > 0) {
      activeLabel.push(`${ipAdapterCount} IP`);
    }

    const controlNetCount = selectAllControlNets(controlAdapters).length;
    if (controlNetCount > 0) {
      activeLabel.push(`${controlNetCount} ControlNet`);
    }

    const t2iAdapterCount = selectAllT2IAdapters(controlAdapters).length;
    if (t2iAdapterCount > 0) {
      activeLabel.push(`${t2iAdapterCount} T2I`);
    }

    const controlAdapterIds =
      selectControlAdapterIds(controlAdapters).map(String);

    return {
      controlAdapterIds,
      activeLabel: activeLabel.join(', '),
    };
  },
  defaultSelectorOptions
);

const ParamControlAdaptersCollapse = () => {
  const { controlAdapterIds, activeLabel } = useAppSelector(selector);
  const isControlNetDisabled = useFeatureStatus('controlNet').isFeatureDisabled;
  const { addControlNet } = useAddControlNet();
  const { addIPAdapter } = useAddIPAdapter();
  const { addT2IAdapter } = useAddT2IAdapter();

  if (isControlNetDisabled) {
    return null;
  }

  return (
    <IAICollapse label="Control Adapters" activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ButtonGroup size="sm" w="full" justifyContent="space-between">
          <IAIButton
            leftIcon={<FaPlus />}
            onClick={addControlNet}
            data-testid="add controlnet"
            flexGrow={1}
          >
            ControlNet
          </IAIButton>
          <IAIButton
            leftIcon={<FaPlus />}
            onClick={addIPAdapter}
            data-testid="add ip adapter"
            flexGrow={1}
          >
            IP Adapter
          </IAIButton>
          <IAIButton
            leftIcon={<FaPlus />}
            onClick={addT2IAdapter}
            data-testid="add t2i adapter"
            flexGrow={1}
          >
            T2I Adapter
          </IAIButton>
        </ButtonGroup>
        <Divider />
        {controlAdapterIds.map((id, i) => (
          <Fragment key={id}>
            {i > 0 && <Divider />}
            <ControlAdapterConfig id={id} />
          </Fragment>
        ))}
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamControlAdaptersCollapse);
