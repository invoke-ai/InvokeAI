import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import type { IPaneviewReactProps } from 'dockview';
import { PaneviewReact } from 'dockview';
import { selectIsCogView4, selectIsSDXL } from 'features/controlLayers/store/paramsSlice';
import { Prompts } from 'features/parameters/components/Prompts/Prompts';
import { useIsApiModel } from 'features/parameters/hooks/useIsApiModel';
import { AdvancedSettingsAccordion } from 'features/settingsAccordions/components/AdvancedSettingsAccordion/AdvancedSettingsAccordion';
import { CompositingSettingsAccordion } from 'features/settingsAccordions/components/CompositingSettingsAccordion/CompositingSettingsAccordion';
import { GenerationSettingsAccordion } from 'features/settingsAccordions/components/GenerationSettingsAccordion/GenerationSettingsAccordion';
import { ImageSettingsAccordion } from 'features/settingsAccordions/components/ImageSettingsAccordion/ImageSettingsAccordion';
import { RefinerSettingsAccordion } from 'features/settingsAccordions/components/RefinerSettingsAccordion/RefinerSettingsAccordion';
import { $isStylePresetsMenuOpen } from 'features/stylePresets/store/stylePresetSlice';
import type { CSSProperties } from 'react';
import { memo } from 'react';

const overlayScrollbarsStyles: CSSProperties = {
  height: '100%',
  width: '100%',
};

const components: IPaneviewReactProps['components'] = {
  prompts: Prompts,
  imageSettings: ImageSettingsAccordion,
  generationSettings: GenerationSettingsAccordion,
  compositingSettings: CompositingSettingsAccordion,
  advancedSettings: AdvancedSettingsAccordion,
  refinerSettings: RefinerSettingsAccordion,
};

const onReady: IPaneviewReactProps['onReady'] = (event) => {
  event.api.addPanel({
    id: 'prompts',
    title: 'Prompts',
    component: 'prompts',
    isExpanded: true,
  });
  event.api.addPanel({
    id: 'imageSettings',
    title: 'Image Settings',
    component: 'imageSettings',
    isExpanded: true,
  });
  event.api.addPanel({
    id: 'generationSettings',
    title: 'Generation Settings',
    component: 'generationSettings',
  });
  event.api.addPanel({
    id: 'compositingSettings',
    title: 'Compositing Settings',
    component: 'compositingSettings',
  });
  event.api.addPanel({
    id: 'advancedSettings',
    title: 'Advanced Settings',
    component: 'advancedSettings',
  });
  event.api.addPanel({
    id: 'refinerSettings',
    title: 'Refiner Settings',
    component: 'refinerSettings',
  });
};

const ParametersPanelTextToImage = () => {
  const isSDXL = useAppSelector(selectIsSDXL);
  const isCogview4 = useAppSelector(selectIsCogView4);
  const isStylePresetsMenuOpen = useStore($isStylePresetsMenuOpen);

  const isApiModel = useIsApiModel();

  return <PaneviewReact components={components} onReady={onReady} />;

  // return (
  //   <Flex w="full" h="full" flexDir="column" gap={2}>
  //     <StylePresetMenuTrigger />
  //     <Flex w="full" h="full" position="relative">
  //       <Box position="absolute" top={0} left={0} right={0} bottom={0}>
  //         {isStylePresetsMenuOpen && (
  //           <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
  //             <Flex gap={2} flexDirection="column" h="full" w="full">
  //               <StylePresetMenu />
  //             </Flex>
  //           </OverlayScrollbarsComponent>
  //         )}
  //         <OverlayScrollbarsComponent defer style={overlayScrollbarsStyles} options={overlayScrollbarsParams.options}>
  //           <Flex gap={2} flexDirection="column" h="full" w="full">
  //             <Prompts />
  //             <ImageSettingsAccordion />
  //             <GenerationSettingsAccordion />
  //             {!isApiModel && <CompositingSettingsAccordion />}
  //             {isSDXL && <RefinerSettingsAccordion />}
  //             {!isCogview4 && !isApiModel && <AdvancedSettingsAccordion />}
  //           </Flex>
  //         </OverlayScrollbarsComponent>
  //       </Box>
  //     </Flex>
  //   </Flex>
  // );
};

export default memo(ParametersPanelTextToImage);
