/* eslint-disable i18next/no-literal-string */
import { Button, Flex } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { AddLayerButton } from 'features/regionalPrompts/components/AddLayerButton';
import { BrushSize } from 'features/regionalPrompts/components/BrushSize';
import { LayerListItem } from 'features/regionalPrompts/components/LayerListItem';
import { RegionalPromptsStage } from 'features/regionalPrompts/components/RegionalPromptsStage';
import { ToolChooser } from 'features/regionalPrompts/components/ToolChooser';
import { selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getRegionalPromptLayerBlobs } from 'features/regionalPrompts/util/getLayerBlobs';
import { ImageSizeLinear } from 'features/settingsAccordions/components/ImageSettingsAccordion/ImageSizeLinear';

const selectLayerIdsReversed = createSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.layers.map((l) => l.id).reverse()
);

const debugBlobs = () => {
  getRegionalPromptLayerBlobs(true);
};

export const RegionalPromptsEditor = () => {
  const layerIdsReversed = useAppSelector(selectLayerIdsReversed);
  return (
    <Flex gap={4}>
      <Flex flexDir="column" gap={4} flexShrink={0}>
        <Button onClick={debugBlobs}>DEBUG LAYERS</Button>
        <AddLayerButton />
        <BrushSize />
        <ImageSizeLinear />
        <ToolChooser />
        {layerIdsReversed.map((id) => (
          <LayerListItem key={id} id={id} />
        ))}
      </Flex>
      <Flex>
        <RegionalPromptsStage />
      </Flex>
    </Flex>
  );
};
