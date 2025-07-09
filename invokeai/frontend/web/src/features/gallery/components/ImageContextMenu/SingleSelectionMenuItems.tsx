import { MenuDivider } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { ImageMenuItemChangeBoard } from 'features/gallery/components/ImageContextMenu/ImageMenuItemChangeBoard';
import { ImageMenuItemCopy } from 'features/gallery/components/ImageContextMenu/ImageMenuItemCopy';
import { ImageMenuItemDelete } from 'features/gallery/components/ImageContextMenu/ImageMenuItemDelete';
import { ImageMenuItemDownload } from 'features/gallery/components/ImageContextMenu/ImageMenuItemDownload';
import { ImageMenuItemLoadWorkflow } from 'features/gallery/components/ImageContextMenu/ImageMenuItemLoadWorkflow';
import { ImageMenuItemMetadataRecallActions } from 'features/gallery/components/ImageContextMenu/ImageMenuItemMetadataRecallActions';
import { ImageMenuItemNewCanvasFromImageSubMenu } from 'features/gallery/components/ImageContextMenu/ImageMenuItemNewCanvasFromImageSubMenu';
import { ImageMenuItemNewLayerFromImageSubMenu } from 'features/gallery/components/ImageContextMenu/ImageMenuItemNewLayerFromImageSubMenu';
import { ImageMenuItemOpenInNewTab } from 'features/gallery/components/ImageContextMenu/ImageMenuItemOpenInNewTab';
import { ImageMenuItemOpenInViewer } from 'features/gallery/components/ImageContextMenu/ImageMenuItemOpenInViewer';
import { ImageMenuItemSelectForCompare } from 'features/gallery/components/ImageContextMenu/ImageMenuItemSelectForCompare';
import { ImageMenuItemSendToUpscale } from 'features/gallery/components/ImageContextMenu/ImageMenuItemSendToUpscale';
import { ImageMenuItemStarUnstar } from 'features/gallery/components/ImageContextMenu/ImageMenuItemStarUnstar';
import { ImageMenuItemUseAsRefImage } from 'features/gallery/components/ImageContextMenu/ImageMenuItemUseAsRefImage';
import { ImageMenuItemUseForPromptGeneration } from 'features/gallery/components/ImageContextMenu/ImageMenuItemUseForPromptGeneration';
import { ImageDTOContextProvider } from 'features/gallery/contexts/ImageDTOContext';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { memo } from 'react';
import type { ImageDTO } from 'services/api/types';

type SingleSelectionMenuItemsProps = {
  imageDTO: ImageDTO;
};

const SingleSelectionMenuItems = ({ imageDTO }: SingleSelectionMenuItemsProps) => {
  const tab = useAppSelector(selectActiveTab);

  return (
    <ImageDTOContextProvider value={imageDTO}>
      <IconMenuItemGroup>
        <ImageMenuItemOpenInNewTab />
        <ImageMenuItemCopy />
        <ImageMenuItemDownload />
        <ImageMenuItemOpenInViewer />
        <ImageMenuItemSelectForCompare />
        <ImageMenuItemDelete />
      </IconMenuItemGroup>
      <MenuDivider />
      <ImageMenuItemLoadWorkflow />
      {(tab === 'canvas' || tab === 'generate') && <ImageMenuItemMetadataRecallActions />}
      <MenuDivider />
      <ImageMenuItemSendToUpscale />
      <ImageMenuItemUseForPromptGeneration />
      {(tab === 'canvas' || tab === 'generate') && <ImageMenuItemUseAsRefImage />}
      <ImageMenuItemNewCanvasFromImageSubMenu />
      {tab === 'canvas' && <ImageMenuItemNewLayerFromImageSubMenu />}
      <MenuDivider />
      <ImageMenuItemChangeBoard />
      <ImageMenuItemStarUnstar />
    </ImageDTOContextProvider>
  );
};

export default memo(SingleSelectionMenuItems);
