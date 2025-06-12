import { Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { CLIPVisionModel } from 'features/controlLayers/components/common/CLIPVisionModel';
import { FLUXReduxImageInfluence } from 'features/controlLayers/components/common/FLUXReduxImageInfluence';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { GlobalReferenceImageModel } from 'features/controlLayers/components/IPAdapter/GlobalReferenceImageModel';
import { IPAdapterMethod } from 'features/controlLayers/components/IPAdapter/IPAdapterMethod';
import { IPAdapterSettingsEmptyState } from 'features/controlLayers/components/IPAdapter/IPAdapterSettingsEmptyState';
import { useRefImageIdContext } from 'features/controlLayers/contexts/RefImageIdContext';
import { selectIsFLUX } from 'features/controlLayers/store/paramsSlice';
import {
  referenceImageIPAdapterBeginEndStepPctChanged,
  referenceImageIPAdapterCLIPVisionModelChanged,
  referenceImageIPAdapterFLUXReduxImageInfluenceChanged,
  referenceImageIPAdapterImageChanged,
  referenceImageIPAdapterMethodChanged,
  referenceImageIPAdapterModelChanged,
  referenceImageIPAdapterWeightChanged,
  selectRefImageEntity,
  selectRefImageEntityOrThrow,
  selectRefImagesSlice,
} from 'features/controlLayers/store/refImagesSlice';
import type {
  CLIPVisionModelV2,
  FLUXReduxImageInfluence as FLUXReduxImageInfluenceType,
  IPMethodV2,
} from 'features/controlLayers/store/types';
import type { SetGlobalReferenceImageDndTargetData } from 'features/dnd/dnd';
import { setGlobalReferenceImageDndTarget } from 'features/dnd/dnd';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import type { ApiModelConfig, FLUXReduxModelConfig, ImageDTO, IPAdapterModelConfig } from 'services/api/types';

import { IPAdapterImagePreview } from './IPAdapterImagePreview';

const buildSelectIPAdapter = (id: string) =>
  createSelector(
    selectRefImagesSlice,
    (refImages) => selectRefImageEntityOrThrow(refImages, id, 'IPAdapterSettings').ipAdapter
  );

const IPAdapterSettingsContent = memo(() => {
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const id = useRefImageIdContext();
  const selectIPAdapter = useMemo(() => buildSelectIPAdapter(id), [id]);
  const ipAdapter = useAppSelector(selectIPAdapter);

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(referenceImageIPAdapterBeginEndStepPctChanged({ id, beginEndStepPct }));
    },
    [dispatch, id]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(referenceImageIPAdapterWeightChanged({ id, weight }));
    },
    [dispatch, id]
  );

  const onChangeIPMethod = useCallback(
    (method: IPMethodV2) => {
      dispatch(referenceImageIPAdapterMethodChanged({ id, method }));
    },
    [dispatch, id]
  );

  const onChangeFLUXReduxImageInfluence = useCallback(
    (imageInfluence: FLUXReduxImageInfluenceType) => {
      dispatch(referenceImageIPAdapterFLUXReduxImageInfluenceChanged({ id, imageInfluence }));
    },
    [dispatch, id]
  );

  const onChangeModel = useCallback(
    (modelConfig: IPAdapterModelConfig | FLUXReduxModelConfig | ApiModelConfig) => {
      dispatch(referenceImageIPAdapterModelChanged({ id, modelConfig }));
    },
    [dispatch, id]
  );

  const onChangeCLIPVisionModel = useCallback(
    (clipVisionModel: CLIPVisionModelV2) => {
      dispatch(referenceImageIPAdapterCLIPVisionModelChanged({ id, clipVisionModel }));
    },
    [dispatch, id]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(referenceImageIPAdapterImageChanged({ id, imageDTO }));
    },
    [dispatch, id]
  );

  const dndTargetData = useMemo<SetGlobalReferenceImageDndTargetData>(
    () => setGlobalReferenceImageDndTarget.getData({ id }, ipAdapter.image?.image_name),
    [id, ipAdapter.image?.image_name]
  );
  // const pullBboxIntoIPAdapter = usePullBboxIntoGlobalReferenceImage(id);
  // const isBusy = useCanvasIsBusy();

  const isFLUX = useAppSelector(selectIsFLUX);

  return (
    <Flex flexDir="column" gap={2} position="relative" w="full">
      <Flex gap={2} alignItems="center" w="full">
        <GlobalReferenceImageModel modelKey={ipAdapter.model?.key ?? null} onChangeModel={onChangeModel} />
        {ipAdapter.type === 'ip_adapter' && (
          <CLIPVisionModel model={ipAdapter.clipVisionModel} onChange={onChangeCLIPVisionModel} />
        )}
        {/* <IconButton
            onClick={pullBboxIntoIPAdapter}
            isDisabled={isBusy}
            variant="ghost"
            aria-label={t('controlLayers.pullBboxIntoReferenceImage')}
            tooltip={t('controlLayers.pullBboxIntoReferenceImage')}
            icon={<PiBoundingBoxBold />}
          /> */}
      </Flex>
      <Flex gap={2} w="full">
        {ipAdapter.type === 'ip_adapter' && (
          <Flex flexDir="column" gap={2} w="full">
            {!isFLUX && <IPAdapterMethod method={ipAdapter.method} onChange={onChangeIPMethod} />}
            <Weight weight={ipAdapter.weight} onChange={onChangeWeight} />
            <BeginEndStepPct beginEndStepPct={ipAdapter.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
          </Flex>
        )}
        {ipAdapter.type === 'flux_redux' && (
          <Flex flexDir="column" gap={2} w="full" alignItems="flex-start">
            <FLUXReduxImageInfluence
              imageInfluence={ipAdapter.imageInfluence ?? 'lowest'}
              onChange={onChangeFLUXReduxImageInfluence}
            />
          </Flex>
        )}
        <Flex alignItems="center" justifyContent="center" h={32} w={32} aspectRatio="1/1" flexGrow={1}>
          <IPAdapterImagePreview
            image={ipAdapter.image}
            onChangeImage={onChangeImage}
            dndTarget={setGlobalReferenceImageDndTarget}
            dndTargetData={dndTargetData}
          />
        </Flex>
      </Flex>
    </Flex>
  );
});

IPAdapterSettingsContent.displayName = 'IPAdapterSettingsContent';

const buildSelectIPAdapterHasImage = (id: string) =>
  createSelector(selectRefImagesSlice, (refImages) => {
    const referenceImage = selectRefImageEntity(refImages, id);
    return !!referenceImage && referenceImage.ipAdapter.image !== null;
  });

export const IPAdapterSettings = memo(() => {
  const id = useRefImageIdContext();

  const selectIPAdapterHasImage = useMemo(() => buildSelectIPAdapterHasImage(id), [id]);
  const hasImage = useAppSelector(selectIPAdapterHasImage);

  if (!hasImage) {
    return <IPAdapterSettingsEmptyState />;
  }

  return <IPAdapterSettingsContent />;
});

IPAdapterSettings.displayName = 'IPAdapterSettings';
