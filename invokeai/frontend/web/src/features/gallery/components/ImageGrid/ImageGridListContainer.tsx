import type { FlexProps } from '@invoke-ai/ui-library';
import { forwardRef, Grid } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

export const imageListContainerTestId = 'image-list-container';

type ListContainerProps = PropsWithChildren & FlexProps;
const ListContainer = forwardRef((props: ListContainerProps, ref) => {
  const galleryImageMinimumWidth = useAppSelector((s) => s.gallery.galleryImageMinimumWidth);

  return (
    <Grid
      {...props}
      className="list-container"
      ref={ref}
      gridTemplateColumns={`repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr))`}
      data-testid={imageListContainerTestId}
    >
      {props.children}
    </Grid>
  );
});

export default memo(ListContainer);
