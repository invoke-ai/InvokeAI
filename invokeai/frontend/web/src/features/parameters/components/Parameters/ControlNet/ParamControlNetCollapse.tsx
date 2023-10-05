import { ButtonGroup, Divider, Flex } from '@chakra-ui/react';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { defaultSelectorOptions } from 'app/store/util/defaultMemoizeOptions';
import IAIButton from 'common/components/IAIButton';
import IAICollapse from 'common/components/IAICollapse';
import ControlNet from 'features/controlNet/components/ControlNet';
import { useAddControlNet } from 'features/controlNet/hooks/useAddControlNet';
import { useAddIPAdapter } from 'features/controlNet/hooks/useAddIPAdapter';
import { useAddT2IAdapter } from 'features/controlNet/hooks/useAddT2IAdapter';
import {
  selectAllControlNets,
  selectAllIPAdapters,
  selectAllT2IAdapters,
} from 'features/controlNet/store/controlAdaptersSlice';
import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { Fragment, memo } from 'react';
import { FaPlus } from 'react-icons/fa';
import { useGetControlNetModelsQuery } from 'services/api/endpoints/models';

const selector = createSelector(
  [stateSelector],
  ({ controlAdapters }) => {
    const activeLabel: string[] = [];

    const validIPAdapters = selectAllIPAdapters(controlAdapters);
    const validIPAdapterCount = validIPAdapters.length;
    if (validIPAdapterCount > 0) {
      activeLabel.push(`${validIPAdapterCount} IP`);
    }

    const validControlNets = selectAllControlNets(controlAdapters);
    const validControlNetCount = validControlNets.length;
    if (validControlNetCount > 0) {
      activeLabel.push(`${validControlNetCount} ControlNet`);
    }

    const validT2IAdapters = selectAllT2IAdapters(controlAdapters);
    const validT2IAdapterCount = validT2IAdapters.length;
    if (validT2IAdapterCount > 0) {
      activeLabel.push(`${validT2IAdapterCount} T2I`);
    }

    return {
      controlAdapters: [
        ...validIPAdapters,
        ...validControlNets,
        ...validT2IAdapters,
      ],
      activeLabel: activeLabel.join(', '),
    };
  },
  defaultSelectorOptions
);

const ParamControlNetCollapse = () => {
  const { controlAdapters, activeLabel } = useAppSelector(selector);
  const isControlNetDisabled = useFeatureStatus('controlNet').isFeatureDisabled;
  const dispatch = useAppDispatch();
  const { data: controlnetModels } = useGetControlNetModelsQuery();
  const { addControlNet } = useAddControlNet();
  const { addIPAdapter } = useAddIPAdapter();
  const { addT2IAdapter } = useAddT2IAdapter();

  return (
    <IAICollapse label="Control Adapters" activeLabel={activeLabel}>
      <Flex sx={{ flexDir: 'column', gap: 2 }}>
        <ButtonGroup size="sm" w="full" justifyContent="space-between">
          <IAIButton
            leftIcon={<FaPlus />}
            onClick={addControlNet}
            data-testid="add controlnet"
          >
            ControlNet
          </IAIButton>
          <IAIButton
            leftIcon={<FaPlus />}
            onClick={addIPAdapter}
            data-testid="add ip adapter"
          >
            IP Adapter
          </IAIButton>
          <IAIButton
            leftIcon={<FaPlus />}
            onClick={addT2IAdapter}
            data-testid="add t2i adapter"
          >
            T2I Adapter
          </IAIButton>
        </ButtonGroup>
        {controlAdapters.map((ca, i) => (
          <Fragment key={ca.id}>
            {i > 0 && <Divider />}
            <ControlNet id={ca.id} />
          </Fragment>
        ))}
      </Flex>
    </IAICollapse>
  );
};

export default memo(ParamControlNetCollapse);
