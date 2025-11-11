import { Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { FLUXReduxImageInfluence } from 'features/controlLayers/components/common/FLUXReduxImageInfluence';
import { IPAdapterCLIPVisionModel } from 'features/controlLayers/components/common/IPAdapterCLIPVisionModel';
import { PullBboxIntoRefImageIconButton } from 'features/controlLayers/components/common/PullBboxIntoRefImageIconButton';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { IPAdapterMethod } from 'features/controlLayers/components/RefImage/IPAdapterMethod';
import { RefImageModel } from 'features/controlLayers/components/RefImage/RefImageModel';
import { RefImageNoImageState } from 'features/controlLayers/components/RefImage/RefImageNoImageState';
import { RefImageNoImageStateWithCanvasOptions } from 'features/controlLayers/components/RefImage/RefImageNoImageStateWithCanvasOptions';
import {
  CanvasManagerProviderGate,
  useCanvasManagerSafe,
} from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import {
  refImageFLUXReduxImageInfluenceChanged,
  refImageImageChanged,
  refImageIPAdapterBeginEndStepPctChanged,
  refImageIPAdapterCLIPVisionModelChanged,
  refImageIPAdapterMethodChanged,
  refImageIPAdapterWeightChanged,
  refImageModelChanged,
  selectRefImageEntity,
  selectRefImageEntityOrThrow,
  selectRefImagesSlice,
} from 'features/controlLayers/store/refImagesSlice';
import { selectActiveTab } from 'features/controlLayers/store/selectors';
import type {
  CLIPVisionModelV2,
  CroppableImageWithDims,
  FLUXReduxImageInfluence as FLUXReduxImageInfluenceType,
  IPMethodV2,
} from 'features/controlLayers/store/types';
import { isFLUXReduxConfig, isIPAdapterConfig } from 'features/controlLayers/store/types';
import type { SetGlobalReferenceImageDndTargetData } from 'features/dnd/dnd';
import { setGlobalReferenceImageDndTarget } from 'features/dnd/dnd';
import { memo, useCallback, useMemo } from 'react';
import type {
  ChatGPT4oModelConfig,
  FLUXKontextModelConfig,
  FLUXReduxModelConfig,
  IPAdapterModelConfig,
} from 'services/api/types';

import { RefImageImage } from './RefImageImage';

const buildSelectConfig = (id: string) =>
  createSelector(
    selectRefImagesSlice,
    (refImages) => selectRefImageEntityOrThrow(refImages, id, 'IPAdapterSettings').config
  );

const RefImageSettingsContent = memo(() => {
  const dispatch = useAppDispatch();
  const id = useRefImageIdContext();
  const selectConfig = useMemo(() => buildSelectConfig(id), [id]);
  const config = useAppSelector(selectConfig);
  const tab = useAppSelector(selectActiveTab);

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(refImageIPAdapterBeginEndStepPctChanged({ id, beginEndStepPct }));
    },
    [dispatch, id]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(refImageIPAdapterWeightChanged({ id, weight }));
    },
    [dispatch, id]
  );

  const onChangeIPMethod = useCallback(
    (method: IPMethodV2) => {
      dispatch(refImageIPAdapterMethodChanged({ id, method }));
    },
    [dispatch, id]
  );

  const onChangeFLUXReduxImageInfluence = useCallback(
    (imageInfluence: FLUXReduxImageInfluenceType) => {
      dispatch(refImageFLUXReduxImageInfluenceChanged({ id, imageInfluence }));
    },
    [dispatch, id]
  );

  const onChangeModel = useCallback(
    (modelConfig: IPAdapterModelConfig | FLUXReduxModelConfig | ChatGPT4oModelConfig | FLUXKontextModelConfig) => {
      dispatch(refImageModelChanged({ id, modelConfig }));
    },
    [dispatch, id]
  );

  const onChangeCLIPVisionModel = useCallback(
    (clipVisionModel: CLIPVisionModelV2) => {
      dispatch(refImageIPAdapterCLIPVisionModelChanged({ id, clipVisionModel }));
    },
    [dispatch, id]
  );

  const onChangeImage = useCallback(
    (croppableImage: CroppableImageWithDims | null) => {
      dispatch(refImageImageChanged({ id, croppableImage }));
    },
    [dispatch, id]
  );

  const dndTargetData = useMemo<SetGlobalReferenceImageDndTargetData>(
    () =>
      setGlobalReferenceImageDndTarget.getData(
        { id },
        config.image?.crop?.image.image_name ?? config.image?.original.image.image_name
      ),
    [id, config.image?.crop?.image.image_name, config.image?.original.image.image_name]
  );

  const isFLUX = useAppSelector(selectIsFLUX);

  return (
    <Flex flexDir="column" gap={2} position="relative" w="full">
      <Flex gap={2} alignItems="center" w="full">
        <RefImageModel modelKey={config.model?.key ?? null} onChangeModel={onChangeModel} />
        {isIPAdapterConfig(config) && (
          <IPAdapterCLIPVisionModel model={config.clipVisionModel} onChange={onChangeCLIPVisionModel} />
        )}
        {tab === 'canvas' && (
          <CanvasManagerProviderGate>
            <PullBboxIntoRefImageIconButton />
          </CanvasManagerProviderGate>
        )}
      </Flex>
      <Flex gap={2} w="full">
        {isIPAdapterConfig(config) && (
          <Flex flexDir="column" gap={2} w="full">
            {!isFLUX && <IPAdapterMethod method={config.method} onChange={onChangeIPMethod} />}
            <Weight weight={config.weight} onChange={onChangeWeight} />
            <BeginEndStepPct beginEndStepPct={config.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
          </Flex>
        )}
        {isFLUXReduxConfig(config) && (
          <Flex flexDir="column" gap={2} w="full" alignItems="flex-start">
            <FLUXReduxImageInfluence
              imageInfluence={config.imageInfluence ?? 'lowest'}
              onChange={onChangeFLUXReduxImageInfluence}
            />
          </Flex>
        )}
        <Flex alignItems="center" justifyContent="center" h={32} w={32} aspectRatio="1/1" flexGrow={1}>
          <RefImageImage
            image={config.image}
            onChangeImage={onChangeImage}
            dndTarget={setGlobalReferenceImageDndTarget}
            dndTargetData={dndTargetData}
          />
        </Flex>
      </Flex>
    </Flex>
  );
});

RefImageSettingsContent.displayName = 'RefImageSettingsContent';

const buildSelectIPAdapterHasImage = (id: string) =>
  createSelector(selectRefImagesSlice, (refImages) => {
    const referenceImage = selectRefImageEntity(refImages, id);
    return !!referenceImage && referenceImage.config.image !== null;
  });

export const RefImageSettings = memo(() => {
  const id = useRefImageIdContext();
  const tab = useAppSelector(selectActiveTab);
  const canvasManager = useCanvasManagerSafe();
  const selectIPAdapterHasImage = useMemo(() => buildSelectIPAdapterHasImage(id), [id]);
  const hasImage = useAppSelector(selectIPAdapterHasImage);

  if (!hasImage && canvasManager && tab === 'canvas') {
    return (
      <CanvasManagerProviderGate>
        <RefImageNoImageStateWithCanvasOptions />
      </CanvasManagerProviderGate>
    );
  }

  if (!hasImage) {
    return <RefImageNoImageState />;
  }

  return <RefImageSettingsContent />;
});

RefImageSettings.displayName = 'RefImageSettings';
