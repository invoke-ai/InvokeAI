import { Flex } from '@invoke-ai/ui-library';
import { useAppDispatch } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { ControlAdapterControlModeSelect } from 'features/controlLayers/components/ControlAdapter/ControlAdapterControlModeSelect';
import { ControlAdapterModel } from 'features/controlLayers/components/ControlAdapter/ControlAdapterModel';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useControlLayerControlAdapter } from 'features/controlLayers/hooks/useLayerControlAdapter';
import {
  controlLayerBeginEndStepPctChanged,
  controlLayerControlModeChanged,
  controlLayerModelChanged,
  controlLayerWeightChanged,
} from 'features/controlLayers/store/canvasV2Slice';
import type { ControlModeV2 } from 'features/controlLayers/store/types';
import { memo, useCallback } from 'react';
import type { ControlNetModelConfig, T2IAdapterModelConfig } from 'services/api/types';

export const ControlLayerControlAdapter = memo(() => {
  const dispatch = useAppDispatch();
  const entityIdentifier = useEntityIdentifierContext();
  const controlAdapter = useControlLayerControlAdapter(entityIdentifier);

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(controlLayerBeginEndStepPctChanged({ id: entityIdentifier.id, beginEndStepPct }));
    },
    [dispatch, entityIdentifier.id]
  );

  const onChangeControlMode = useCallback(
    (controlMode: ControlModeV2) => {
      dispatch(controlLayerControlModeChanged({ id: entityIdentifier.id, controlMode }));
    },
    [dispatch, entityIdentifier.id]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(controlLayerWeightChanged({ id: entityIdentifier.id, weight }));
    },
    [dispatch, entityIdentifier.id]
  );

  const onChangeModel = useCallback(
    (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig) => {
      dispatch(controlLayerModelChanged({ id: entityIdentifier.id, modelConfig }));
    },
    [dispatch, entityIdentifier.id]
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

ControlLayerControlAdapter.displayName = 'ControlLayerControlAdapter';
