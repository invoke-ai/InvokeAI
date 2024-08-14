import { Flex } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { ControlAdapterControlModeSelect } from 'features/controlLayers/components/ControlAdapter/ControlAdapterControlModeSelect';
import { ControlAdapterModel } from 'features/controlLayers/components/ControlAdapter/ControlAdapterModel';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  layerControlAdapterBeginEndStepPctChanged,
  layerControlAdapterControlModeChanged,
  layerControlAdapterModelChanged,
  layerControlAdapterWeightChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import type { ControlModeV2, ControlNetConfig, T2IAdapterConfig } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import type { ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';

type Props = {
  controlAdapter: ControlNetConfig | T2IAdapterConfig;
};

export const LayerControlAdapter = memo(({ controlAdapter }: Props) => {
  const dispatch = useAppDispatch();
  const { id } = useEntityIdentifierContext();

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(layerControlAdapterBeginEndStepPctChanged({ id, beginEndStepPct }));
    },
    [dispatch, id]
  );

  const onChangeControlMode = useCallback(
    (controlMode: ControlModeV2) => {
      dispatch(layerControlAdapterControlModeChanged({ id, controlMode }));
    },
    [dispatch, id]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(layerControlAdapterWeightChanged({ id, weight }));
    },
    [dispatch, id]
  );

  const onChangeModel = useCallback(
    (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig) => {
      dispatch(layerControlAdapterModelChanged({ id, modelConfig }));
    },
    [dispatch, id]
  );

  return (
    <Flex flexDir="column" gap={3} position="relative" w="full">
      <ControlAdapterModel modelKey={controlAdapter.model?.key ?? null} onChange={onChangeModel} />
      <Weight weight={controlAdapter.weight} onChange={onChangeWeight} />
      <BeginEndStepPct beginEndStepPct={controlAdapter.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
      {controlAdapter.type === 'controlnet' && (
        <ControlAdapterControlModeSelect controlMode={controlAdapter.controlMode} onChange={onChangeControlMode} />
      )}
    </Flex>
  );
});

LayerControlAdapter.displayName = 'LayerControlAdapter';
