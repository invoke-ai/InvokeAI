import { Box, Flex, Icon, IconButton } from '@invoke-ai/ui-library';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { ControlAdapterModelCombobox } from 'features/controlLayers/components/CALayer/ControlAdapterModelCombobox';
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
import { memo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { PiCaretUpBold } from 'react-icons/pi';
import { useToggle } from 'react-use';
import type { ControlNetModelConfig, ImageDTO, T2IAdapterModelConfig } from 'services/api/types';

import { CALayerImagePreview } from './CALayerImagePreview';
import { CALayerProcessor } from './CALayerProcessor';
import { CALayerProcessorCombobox } from './CALayerProcessorCombobox';
import { ControlAdapterBeginEndStepPct } from './ControlAdapterBeginEndStepPct';
import { ControlAdapterControlModeSelect } from './ControlAdapterControlModeSelect';
import { ControlAdapterWeight } from './ControlAdapterWeight';

type Props = {
  layerId: string;
};

export const CALayerConfig = memo(({ layerId }: Props) => {
  const dispatch = useAppDispatch();
  const controlAdapter = useAppSelector((s) => selectCALayer(s.controlLayers.present, layerId).controlAdapter);
  const { t } = useTranslation();
  const [isExpanded, toggleIsExpanded] = useToggle(false);

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

  return (
    <Flex flexDir="column" gap={4} position="relative" w="full">
      <Flex gap={3} alignItems="center" w="full">
        <Box minW={0} w="full" transitionProperty="common" transitionDuration="0.1s">
          <ControlAdapterModelCombobox modelKey={controlAdapter.model?.key ?? null} onChange={onChangeModel} />
        </Box>

        <IconButton
          size="sm"
          tooltip={isExpanded ? t('controlnet.hideAdvanced') : t('controlnet.showAdvanced')}
          aria-label={isExpanded ? t('controlnet.hideAdvanced') : t('controlnet.showAdvanced')}
          onClick={toggleIsExpanded}
          variant="ghost"
          icon={
            <Icon
              boxSize={4}
              as={PiCaretUpBold}
              transform={isExpanded ? 'rotate(0deg)' : 'rotate(180deg)'}
              transitionProperty="common"
              transitionDuration="normal"
            />
          }
        />
      </Flex>
      <Flex gap={4} w="full" alignItems="center">
        <Flex flexDir="column" gap={3} w="full">
          {controlAdapter.type === 'controlnet' && (
            <ControlAdapterControlModeSelect controlMode={controlAdapter.controlMode} onChange={onChangeControlMode} />
          )}
          <ControlAdapterWeight weight={controlAdapter.weight} onChange={onChangeWeight} />
          <ControlAdapterBeginEndStepPct
            beginEndStepPct={controlAdapter.beginEndStepPct}
            onChange={onChangeBeginEndStepPct}
          />
        </Flex>
        <Flex alignItems="center" justifyContent="center" h={36} w={36} aspectRatio="1/1">
          <CALayerImagePreview
            image={controlAdapter.image}
            processedImage={controlAdapter.processedImage}
            onChangeImage={onChangeImage}
            layerId={layerId}
            hasProcessor={Boolean(controlAdapter.processorConfig)}
          />
        </Flex>
      </Flex>
      {isExpanded && (
        <>
          <CALayerProcessorCombobox config={controlAdapter.processorConfig} onChange={onChangeProcessorConfig} />
          <CALayerProcessor config={controlAdapter.processorConfig} onChange={onChangeProcessorConfig} />
        </>
      )}
    </Flex>
  );
});

CALayerConfig.displayName = 'CALayerConfig';
