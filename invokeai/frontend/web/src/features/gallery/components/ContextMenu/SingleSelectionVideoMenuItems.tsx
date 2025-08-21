import { MenuDivider } from '@invoke-ai/ui-library';
import { IconMenuItemGroup } from 'common/components/IconMenuItem';
import { ContextMenuItemChangeBoard } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemChangeBoard';
import { ContextMenuItemCopy } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemCopy';
import { ContextMenuItemDelete } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemDelete';
import { ContextMenuItemDownload } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemDownload';
import { ContextMenuItemOpenInNewTab } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemOpenInNewTab';
import { ContextMenuItemOpenInViewer } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemOpenInViewer';
import { ContextMenuItemSelectForCompare } from 'features/gallery/components/ContextMenu/MenuItems/ContextMenuItemSelectForCompare';
import { ItemDTOContextProvider } from 'features/gallery/contexts/ItemDTOContext';
import type { VideoDTO } from 'services/api/types';

import { ContextMenuItemStarUnstar } from './MenuItems/ContextMenuItemStarUnstar';

type SingleSelectionVideoMenuItemsProps = {
  videoDTO: VideoDTO;
};

const SingleSelectionVideoMenuItems = ({ videoDTO }: SingleSelectionVideoMenuItemsProps) => {
  return (
    <ItemDTOContextProvider value={videoDTO}>
      <IconMenuItemGroup>
        <ContextMenuItemOpenInNewTab />
        <ContextMenuItemCopy />
        <ContextMenuItemDownload />
        <ContextMenuItemOpenInViewer />
        <ContextMenuItemSelectForCompare />
        <ContextMenuItemDelete />
      </IconMenuItemGroup>
      <MenuDivider />
      <ContextMenuItemStarUnstar />
      <ContextMenuItemChangeBoard />
    </ItemDTOContextProvider>
  );
};

export default SingleSelectionVideoMenuItems;
