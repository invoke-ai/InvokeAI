import { Box, Flex, IconButton, Spacer, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { BeginEndStepPct } from 'features/controlLayers/components/common/BeginEndStepPct';
import { Weight } from 'features/controlLayers/components/common/Weight';
import { IPAdapterImagePreview } from 'features/controlLayers/components/IPAdapter/IPAdapterImagePreview';
import { IPAdapterMethod } from 'features/controlLayers/components/IPAdapter/IPAdapterMethod';
import { IPAdapterModel } from 'features/controlLayers/components/IPAdapter/IPAdapterModel';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import {
  useIsSavingCanvas,
  usePullBboxIntoRegionalGuidanceIPAdapter,
} from 'features/controlLayers/hooks/saveCanvasHooks';
import {
  rgIPAdapterBeginEndStepPctChanged,
  rgIPAdapterCLIPVisionModelChanged,
  rgIPAdapterDeleted,
  rgIPAdapterImageChanged,
  rgIPAdapterMethodChanged,
  rgIPAdapterModelChanged,
  rgIPAdapterWeightChanged,
} from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectRegionalGuidanceIPAdapter } from 'features/controlLayers/store/selectors';
import type { CLIPVisionModelV2, IPMethodV2 } from 'features/controlLayers/store/types';
import type { RGIPAdapterImageDropData } from 'features/dnd/types';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiBoundingBoxBold, PiTrashSimpleBold } from 'react-icons/pi';
import type { ImageDTO, IPAdapterModelConfig, RGIPAdapterImagePostUploadAction } from 'services/api/types';
import { assert } from 'tsafe';

type Props = {
  ipAdapterId: string;
  ipAdapterNumber: number;
};

export const RegionalGuidanceIPAdapterSettings = memo(({ ipAdapterId, ipAdapterNumber }: Props) => {
  const entityIdentifier = useEntityIdentifierContext('regional_guidance');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();
  const onDeleteIPAdapter = useCallback(() => {
    dispatch(rgIPAdapterDeleted({ entityIdentifier, ipAdapterId }));
  }, [dispatch, entityIdentifier, ipAdapterId]);
  const selectIPAdapter = useMemo(
    () =>
      createSelector(selectCanvasSlice, (canvas) => {
        const ipAdapter = selectRegionalGuidanceIPAdapter(canvas, entityIdentifier, ipAdapterId);
        assert(ipAdapter, `Regional GuidanceIP Adapter with id ${ipAdapterId} not found`);
        return ipAdapter;
      }),
    [entityIdentifier, ipAdapterId]
  );
  const ipAdapter = useAppSelector(selectIPAdapter);

  const onChangeBeginEndStepPct = useCallback(
    (beginEndStepPct: [number, number]) => {
      dispatch(rgIPAdapterBeginEndStepPctChanged({ entityIdentifier, ipAdapterId, beginEndStepPct }));
    },
    [dispatch, entityIdentifier, ipAdapterId]
  );

  const onChangeWeight = useCallback(
    (weight: number) => {
      dispatch(rgIPAdapterWeightChanged({ entityIdentifier, ipAdapterId, weight }));
    },
    [dispatch, entityIdentifier, ipAdapterId]
  );

  const onChangeIPMethod = useCallback(
    (method: IPMethodV2) => {
      dispatch(rgIPAdapterMethodChanged({ entityIdentifier, ipAdapterId, method }));
    },
    [dispatch, entityIdentifier, ipAdapterId]
  );

  const onChangeModel = useCallback(
    (modelConfig: IPAdapterModelConfig) => {
      dispatch(rgIPAdapterModelChanged({ entityIdentifier, ipAdapterId, modelConfig }));
    },
    [dispatch, entityIdentifier, ipAdapterId]
  );

  const onChangeCLIPVisionModel = useCallback(
    (clipVisionModel: CLIPVisionModelV2) => {
      dispatch(rgIPAdapterCLIPVisionModelChanged({ entityIdentifier, ipAdapterId, clipVisionModel }));
    },
    [dispatch, entityIdentifier, ipAdapterId]
  );

  const onChangeImage = useCallback(
    (imageDTO: ImageDTO | null) => {
      dispatch(rgIPAdapterImageChanged({ entityIdentifier, ipAdapterId, imageDTO }));
    },
    [dispatch, entityIdentifier, ipAdapterId]
  );

  const droppableData = useMemo<RGIPAdapterImageDropData>(
    () => ({
      actionType: 'SET_RG_IP_ADAPTER_IMAGE',
      context: { id: entityIdentifier.id, ipAdapterId },
      id: entityIdentifier.id,
    }),
    [entityIdentifier.id, ipAdapterId]
  );
  const postUploadAction = useMemo<RGIPAdapterImagePostUploadAction>(
    () => ({ type: 'SET_RG_IP_ADAPTER_IMAGE', id: entityIdentifier.id, ipAdapterId }),
    [entityIdentifier.id, ipAdapterId]
  );
  const pullBboxIntoIPAdapter = usePullBboxIntoRegionalGuidanceIPAdapter(entityIdentifier, ipAdapterId);
  const isSaving = useIsSavingCanvas();

  return (
    <Flex flexDir="column" gap={3}>
      <Flex alignItems="center" gap={3}>
        <Text fontWeight="semibold" color="base.400">{`IP Adapter ${ipAdapterNumber}`}</Text>
        <Spacer />
        <IconButton
          size="sm"
          icon={<PiTrashSimpleBold />}
          aria-label="Delete IP Adapter"
          onClick={onDeleteIPAdapter}
          variant="ghost"
          colorScheme="error"
        />
      </Flex>
      <Flex flexDir="column" gap={4} position="relative" w="full">
        <Flex gap={3} alignItems="center" w="full">
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
            isLoading={isSaving.isTrue}
            variant="ghost"
            aria-label={t('controlLayers.pullBboxIntoIPAdapter')}
            tooltip={t('controlLayers.pullBboxIntoIPAdapter')}
            icon={<PiBoundingBoxBold />}
          />
        </Flex>
        <Flex gap={4} w="full" alignItems="center">
          <Flex flexDir="column" gap={3} w="full">
            <IPAdapterMethod method={ipAdapter.method} onChange={onChangeIPMethod} />
            <Weight weight={ipAdapter.weight} onChange={onChangeWeight} />
            <BeginEndStepPct beginEndStepPct={ipAdapter.beginEndStepPct} onChange={onChangeBeginEndStepPct} />
          </Flex>
          <Flex alignItems="center" justifyContent="center" h={36} w={36} aspectRatio="1/1">
            <IPAdapterImagePreview
              image={ipAdapter.image ?? null}
              onChangeImage={onChangeImage}
              ipAdapterId={ipAdapter.id}
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
