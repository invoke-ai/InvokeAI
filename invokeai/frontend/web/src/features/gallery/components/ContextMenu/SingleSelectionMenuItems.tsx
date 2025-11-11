import { MenuDivider } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { selectActiveTab } from 'features/controlLayers/store/selectors';
import { ContextMenuItemChangeBoard } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemChangeBoard';
import { ContextMenuItemCopy } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemCopy';
import { ContextMenuItemDownload } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemDownload';
import { ContextMenuItemLoadWorkflow } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemLoadWorkflow';
import { ContextMenuItemLocateInGalery } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemLocateInGalery';
import { ContextMenuItemMetadataRecallActionsCanvasGenerateTabs } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemMetadataRecallActionsCanvasGenerateTabs';
import { ContextMenuItemNewCanvasFromImageSubMenu } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemNewCanvasFromImageSubMenu';
import { ContextMenuItemNewLayerFromImageSubMenu } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemNewLayerFromImageSubMenu';
import { ContextMenuItemOpenInNewTab } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemOpenInNewTab';
import { ContextMenuItemOpenInViewer } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemOpenInViewer';
import { ContextMenuItemSelectForCompare } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemSelectForCompare';
import { ContextMenuItemSendToUpscale } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemSendToUpscale';
import { ContextMenuItemStarUnstar } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemStarUnstar';
import { ContextMenuItemUseAsPromptTemplate } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemUseAsPromptTemplate';
import { ContextMenuItemUseAsRefImage } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemUseAsRefImage';
import { ImageDTOContextProvider } from 'features/gallery/contexts/ImageDTOContext';
import type { ImageDTO } from 'services/api/types';

import { ContextMenuItemDeleteImage } from './MenuItems/ContextMenuItemDeleteImage';
import { ContextMenuItemMetadataRecallActionsUpscaleTab } from './MenuItems/ContextMenuItemMetadataRecallActionsUpscaleTab';

type SingleSelectionMenuItemsProps = {
  imageDTO: ImageDTO;
};

const SingleSelectionMenuItems = ({ imageDTO }: SingleSelectionMenuItemsProps) => {
  const tab = useAppSelector(selectActiveTab);

  return (
    <ImageDTOContextProvider value={imageDTO}>
      <IconMenuItemGroup>
        <ContextMenuItemOpenInNewTab />
        <ContextMenuItemCopy />
        <ContextMenuItemDownload />
        <ContextMenuItemOpenInViewer />
        <ContextMenuItemSelectForCompare />
        <ContextMenuItemDeleteImage />
      </IconMenuItemGroup>
      <MenuDivider />
      <ContextMenuItemLoadWorkflow />
      {(tab === 'canvas' || tab === 'generate') && <ContextMenuItemMetadataRecallActionsCanvasGenerateTabs />}
      {tab === 'upscaling' && <ContextMenuItemMetadataRecallActionsUpscaleTab />}
      <MenuDivider />
      <ContextMenuItemSendToUpscale />
      {(tab === 'canvas' || tab === 'generate') && <ContextMenuItemUseAsRefImage />}
      <ContextMenuItemUseAsPromptTemplate />
      <ContextMenuItemNewCanvasFromImageSubMenu />
      {tab === 'canvas' && <ContextMenuItemNewLayerFromImageSubMenu />}
      <MenuDivider />
      <ContextMenuItemChangeBoard />
      <ContextMenuItemStarUnstar />
      {(tab === 'canvas' || tab === 'generate' || tab === 'workflows' || tab === 'upscaling') &&
        !imageDTO.is_intermediate && (
          // Only render this button on tabs with a gallery.
          <ContextMenuItemLocateInGalery />
        )}
    </ImageDTOContextProvider>
  );
};

export default SingleSelectionMenuItems;
