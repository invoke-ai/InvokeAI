import { useAppSelector } from 'app/store/storeHooks';
import type { InvContextMenuProps } from 'common/components/InvContextMenu/InvContextMenu';
import { InvContextMenu } from 'common/components/InvContextMenu/InvContextMenu';
import { InvMenuList } from 'common/components/InvMenu/InvMenuList';
import { memo, useCallback } from 'react';
import type { ImageDTO } from 'services/api/types';

import MultipleSelectionMenuItems from './MultipleSelectionMenuItems';
import SingleSelectionMenuItems from './SingleSelectionMenuItems';

type Props = {
  imageDTO: ImageDTO | undefined;
  children: InvContextMenuProps<HTMLDivElement>['children'];
};

const ImageContextMenu = ({ imageDTO, children }: Props) => {
  const selectionCount = useAppSelector((s) => s.gallery.selection.length);

  const renderMenuFunc = useCallback(() => {
    if (!imageDTO) {
      return null;
    }

    if (selectionCount > 1) {
      return (
        <InvMenuList visibility="visible">
          <MultipleSelectionMenuItems />
        </InvMenuList>
      );
    }

    return (
      <InvMenuList visibility="visible">
        <SingleSelectionMenuItems imageDTO={imageDTO} />
      </InvMenuList>
    );
  }, [imageDTO, selectionCount]);

  return (
    <InvContextMenu renderMenu={renderMenuFunc}>{children}</InvContextMenu>
  );
};

export default memo(ImageContextMenu);
