/* eslint-disable i18next/no-literal-string */
import { Button, Flex } from '@invoke-ai/ui-library';
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { AddLayerButton } from 'features/regionalPrompts/components/AddLayerButton';
import { BrushSize } from 'features/regionalPrompts/components/BrushSize';
import { DeleteAllLayersButton } from 'features/regionalPrompts/components/DeleteAllLayersButton';
import { StageComponent } from 'features/regionalPrompts/components/imperative/konvaApiDraft';
import { LayerListItem } from 'features/regionalPrompts/components/LayerListItem';
import { PromptLayerOpacity } from 'features/regionalPrompts/components/PromptLayerOpacity';
import { ToolChooser } from 'features/regionalPrompts/components/ToolChooser';
import { selectRegionalPromptsSlice } from 'features/regionalPrompts/store/regionalPromptsSlice';
import { getRegionalPromptLayerBlobs } from 'features/regionalPrompts/util/getLayerBlobs';
import { ImageSizeLinear } from 'features/settingsAccordions/components/ImageSettingsAccordion/ImageSizeLinear';
import { memo } from 'react';

const selectLayerIdsReversed = createMemoizedSelector(selectRegionalPromptsSlice, (regionalPrompts) =>
  regionalPrompts.layers.map((l) => l.id).reverse()
);

const debugBlobs = () => {
  getRegionalPromptLayerBlobs(undefined, true);
};

export const RegionalPromptsEditor = memo(() => {
  const layerIdsReversed = useAppSelector(selectLayerIdsReversed);
  return (
    <Flex gap={4} w="full" h="full">
      <Flex flexDir="column" gap={4} flexShrink={0}>
        <Flex>
          <Button onClick={debugBlobs}>DEBUG</Button>
          <AddLayerButton />
          <DeleteAllLayersButton />
        </Flex>
        <BrushSize />
        <PromptLayerOpacity />
        <ImageSizeLinear />
        <ToolChooser />
        {layerIdsReversed.map((id) => (
          <LayerListItem key={id} id={id} />
        ))}
      </Flex>
      <Flex>
        <StageComponent />
        {/* <RegionalPromptsStage /> */}
      </Flex>
    </Flex>
  );
});

RegionalPromptsEditor.displayName = 'RegionalPromptsEditor';
