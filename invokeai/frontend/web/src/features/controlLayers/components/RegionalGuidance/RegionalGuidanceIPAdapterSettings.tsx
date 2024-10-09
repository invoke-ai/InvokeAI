import { Box, Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { IPAdapterImagePreview } from 'features/controlLayers/components/IPAdapter/IPAdapterImagePreview';
import { IPAdapterMethod } from 'features/controlLayers/components/IPAdapter/IPAdapterMethod';
import { IPAdapterModel } from 'features/controlLayers/components/IPAdapter/IPAdapterModel';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { usePullBboxIntoRegionalGuidanceReferenceImage } from 'features/controlLayers/hooks/saveCanvasHooks';
import { useCanvasIsBusy } from 'features/controlLayers/hooks/useCanvasIsBusy';
import {
  rgIPAdapterBeginEndStepPctChanged,
  rgIPAdapterCLIPVisionModelChanged,
  rgIPAdapterDeleted,
  rgIPAdapterImageChanged,
  rgIPAdapterMethodChanged,
  rgIPAdapterModelChanged,
  rgIPAdapterWeightChanged,
} from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectRegionalGuidanceReferenceImage } from 'features/controlLayers/store/selectors';
import type { CLIPVisionModelV2, IPMethodV2 } from 'features/controlLayers/store/types';
import type { RGIPAdapterImageDropData } from 'features/dnd/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold, PiTrashSimpleFill } from 'react-icons/pi';
import type { ImageDTO, IPAdapterModelConfig, RGIPAdapterImagePostUploadAction } from 'services/api/types';
import { assert } from 'tsafe';

type Props = {
  referenceImageId: string;
};

export const RegionalGuidanceIPAdapterSettings = memo(({ referenceImageId }: Props) => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onDeleteIPAdapter = useCallback(() => {
    dispatch(rgIPAdapterDeleted({ entityIdentifier, referenceImageId }));
  }, [dispatch, entityIdentifier, referenceImageId]);
  const selectIPAdapter = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => {
        const referenceImage = selectRegionalGuidanceReferenceImage(canvas, entityIdentifier, referenceImageId);
        assert(referenceImage, `Regional Guidance IP Adapter with id ${referenceImageId} not found`);
        return referenceImage.ipAdapter;
      }),
    [entityIdentifier, referenceImageId]
  );
  const ipAdapter = useAppSelector(selectIPAdapter);

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(rgIPAdapterBeginEndStepPctChanged({ entityIdentifier, referenceImageId, beginEndStepPct }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(rgIPAdapterWeightChanged({ entityIdentifier, referenceImageId, weight }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeIPMethod = useCallback(
    (method: IPMethodV2) => {
      dispatch(rgIPAdapterMethodChanged({ entityIdentifier, referenceImageId, method }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeModel = useCallback(
    (modelConfig: IPAdapterModelConfig) => {
      dispatch(rgIPAdapterModelChanged({ entityIdentifier, referenceImageId, modelConfig }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeCLIPVisionModel = useCallback(
    (clipVisionModel: CLIPVisionModelV2) => {
      dispatch(rgIPAdapterCLIPVisionModelChanged({ entityIdentifier, referenceImageId, clipVisionModel }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(rgIPAdapterImageChanged({ entityIdentifier, referenceImageId, imageDTO }));
    },
    [dispatch, entityIdentifier, referenceImageId]
  );

  const droppableData = useMemo<RGIPAdapterImageDropData>(
    () => ({
      actionType: 'SET_RG_IP_ADAPTER_IMAGE',
      context: { id: entityIdentifier.id, referenceImageId: referenceImageId },
      id: entityIdentifier.id,
    }),
    [entityIdentifier.id, referenceImageId]
  );
  const postUploadAction = useMemo<RGIPAdapterImagePostUploadAction>(
    () => ({ type: 'SET_RG_IP_ADAPTER_IMAGE', id: entityIdentifier.id, referenceImageId: referenceImageId }),
    [entityIdentifier.id, referenceImageId]
  );
  const pullBboxIntoIPAdapter = usePullBboxIntoRegionalGuidanceReferenceImage(entityIdentifier, referenceImageId);
  const isBusy = useCanvasIsBusy();

  return (
    <Flex flexDir="column" gap={2}>
      <Flex alignItems="center" gap={2}>
        <Text fontWeight="semibold" color="base.400">
          {t('controlLayers.referenceImage')}
        </Text>
        <Spacer />
        <IconButton
          size="sm"
          variant="link"
          alignSelf="stretch"
          icon={<PiTrashSimpleFill />}
          tooltip={t('controlLayers.deleteReferenceImage')}
          aria-label={t('controlLayers.deleteReferenceImage')}
          onClick={onDeleteIPAdapter}
          colorScheme="error"
        />
      </Flex>
      <Flex flexDir="column" gap={2} position="relative" w="full">
        <Flex gap={2} alignItems="center" w="full">
          <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
            <IPAdapterModel
              modelKey={ipAdapter.model?.key ?? null}
              onChangeModel={onChangeModel}
              clipVisionModel={ipAdapter.clipVisionModel}
              onChangeCLIPVisionModel={onChangeCLIPVisionModel}
            />
          </Box>
          <IconButton
            onClick={pullBboxIntoIPAdapter}
            isDisabled={isBusy}
            variant="ghost"
            aria-label={t('controlLayers.pullBboxIntoReferenceImage')}
            tooltip={t('controlLayers.pullBboxIntoReferenceImage')}
            icon={<PiBoundingBoxBold />}
          />
        </Flex>
        <Flex gap={2} w="full">
          <Flex flexDir="column" gap={2} w="full">
            <IPAdapterMethod method={ipAdapter.method} onChange={onChangeIPMethod} />
            <Weight weight={ipAdapter.weight} onChange={onChangeWeight} />
            <BeginEndStepPct beginEndStepPct={ipAdapter.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
          </Flex>
          <Flex alignItems="center" justifyContent="center" h={32} w={32} aspectRatio="1/1">
            <IPAdapterImagePreview
              image={ipAdapter.image ?? null}
              onChangeImage={onChangeImage}
              droppableData={droppableData}
              postUploadAction={postUploadAction}
            />
          </Flex>
        </Flex>
      </Flex>
    </Flex>
  );
});

RegionalGuidanceIPAdapterSettings.displayName = 'RegionalGuidanceIPAdapterSettings';
