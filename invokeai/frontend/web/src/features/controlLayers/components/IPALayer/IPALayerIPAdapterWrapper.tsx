import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { IPAdapter } from 'features/controlLayers/components/ControlAndIPAdapter/IPAdapter';
import {
  caOrIPALayerBeginEndStepPctChanged,
  caOrIPALayerWeightChanged,
  ipaLayerCLIPVisionModelChanged,
  ipaLayerImageChanged,
  ipaLayerMethodChanged,
  ipaLayerModelChanged,
  selectIPALayer,
} from 'features/controlLayers/store/controlLayersSlice';
import type { CLIPVisionModel, IPMethod } from 'features/controlLayers/util/controlAdapters';
import type { IPALayerImageDropData } from 'features/dnd/types';
import { memo, useCallback, useMemo } from 'react';
import type { ImageDTO, IPAdapterModelConfig, IPALayerImagePostUploadAction } from 'services/api/types';

type Props = {
  layerId: string;
};

export const IPALayerIPAdapterWrapper = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const ipAdapter = useAppSelector((s) => selectIPALayer(s.controlLayers.present, layerId).ipAdapter);

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

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(caOrIPALayerWeightChanged({ layerId, weight }));
    },
    [dispatch, layerId]
  );

  const onChangeIPMethod = useCallback(
    (method: IPMethod) => {
      dispatch(ipaLayerMethodChanged({ layerId, method }));
    },
    [dispatch, layerId]
  );

  const onChangeModel = useCallback(
    (modelConfig: IPAdapterModelConfig) => {
      dispatch(ipaLayerModelChanged({ layerId, modelConfig }));
    },
    [dispatch, layerId]
  );

  const onChangeCLIPVisionModel = useCallback(
    (clipVisionModel: CLIPVisionModel) => {
      dispatch(ipaLayerCLIPVisionModelChanged({ layerId, clipVisionModel }));
    },
    [dispatch, layerId]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(ipaLayerImageChanged({ layerId, imageDTO }));
    },
    [dispatch, layerId]
  );

  const droppableData = useMemo<IPALayerImageDropData>(
    () => ({
      actionType: 'SET_IPA_LAYER_IMAGE',
      context: {
        layerId,
      },
      id: layerId,
    }),
    [layerId]
  );

  const postUploadAction = useMemo<IPALayerImagePostUploadAction>(
    () => ({
      type: 'SET_IPA_LAYER_IMAGE',
      layerId,
    }),
    [layerId]
  );

  return (
    <IPAdapter
      ipAdapter={ipAdapter}
      onChangeBeginEndStepPct={onChangeBeginEndStepPct}
      onChangeWeight={onChangeWeight}
      onChangeIPMethod={onChangeIPMethod}
      onChangeModel={onChangeModel}
      onChangeCLIPVisionModel={onChangeCLIPVisionModel}
      onChangeImage={onChangeImage}
      droppableData={droppableData}
      postUploadAction={postUploadAction}
    />
  );
});

IPALayerIPAdapterWrapper.displayName = 'IPALayerIPAdapterWrapper';
