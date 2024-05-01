import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ControlAdapter } from 'features/controlLayers/components/ControlAndIPAdapter/ControlAdapter';
import {
  caLayerControlModeChanged,
  caLayerImageChanged,
  caLayerModelChanged,
  caLayerProcessorConfigChanged,
  caOrIPALayerBeginEndStepPctChanged,
  caOrIPALayerWeightChanged,
  selectCALayer,
} from 'features/controlLayers/store/controlLayersSlice';
import type { ControlMode, ProcessorConfig } from 'features/controlLayers/util/controlAdapters';
import type { CALayerImageDropData } from 'features/dnd/types';
import { memo, useCallback, useMemo } from 'react';
import type {
  CALayerImagePostUploadAction,
  ControlNetModelConfig,
  ImageDTO,
  T2IAdapterModelConfig,
} from 'services/api/types';

type Props = {
  layerId: string;
};

export const CALayerControlAdapterWrapper = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const controlAdapter = useAppSelector((s) => selectCALayer(s.controlLayers.present, layerId).controlAdapter);

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(
        caOrIPALayerBeginEndStepPctChanged({
          layerId,
          beginEndStepPct,
        })
      );
    },
    [dispatch, layerId]
  );

  const onChangeControlMode = useCallback(
    (controlMode: ControlMode) => {
      dispatch(
        caLayerControlModeChanged({
          layerId,
          controlMode,
        })
      );
    },
    [dispatch, layerId]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(caOrIPALayerWeightChanged({ layerId, weight }));
    },
    [dispatch, layerId]
  );

  const onChangeProcessorConfig = useCallback(
    (processorConfig: ProcessorConfig | null) => {
      dispatch(caLayerProcessorConfigChanged({ layerId, processorConfig }));
    },
    [dispatch, layerId]
  );

  const onChangeModel = useCallback(
    (modelConfig: ControlNetModelConfig | T2IAdapterModelConfig) => {
      dispatch(
        caLayerModelChanged({
          layerId,
          modelConfig,
        })
      );
    },
    [dispatch, layerId]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(caLayerImageChanged({ layerId, imageDTO }));
    },
    [dispatch, layerId]
  );

  const droppableData = useMemo<CALayerImageDropData>(
    () => ({
      actionType: 'SET_CA_LAYER_IMAGE',
      context: {
        layerId,
      },
      id: layerId,
    }),
    [layerId]
  );

  const postUploadAction = useMemo<CALayerImagePostUploadAction>(
    () => ({
      layerId,
      type: 'SET_CA_LAYER_IMAGE',
    }),
    [layerId]
  );

  return (
    <ControlAdapter
      controlAdapter={controlAdapter}
      onChangeBeginEndStepPct={onChangeBeginEndStepPct}
      onChangeControlMode={onChangeControlMode}
      onChangeWeight={onChangeWeight}
      onChangeProcessorConfig={onChangeProcessorConfig}
      onChangeModel={onChangeModel}
      onChangeImage={onChangeImage}
      droppableData={droppableData}
      postUploadAction={postUploadAction}
    />
  );
});

CALayerControlAdapterWrapper.displayName = 'CALayerControlAdapterWrapper';
