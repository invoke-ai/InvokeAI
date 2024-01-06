import type { FlexProps } from '@chakra-ui/react';
import { forwardRef, Grid } from '@chakra-ui/react';
import { useAppSelector } from 'app/store/storeHooks';
import type { PropsWithChildren } from 'react';
import { memo } from 'react';

type ListContainerProps = PropsWithChildren & FlexProps;
const ListContainer = forwardRef((props: ListContainerProps, ref) => {
  const galleryImageMinimumWidth = useAppSelector(
    (s) => s.gallery.galleryImageMinimumWidth
  );

  return (
    <Grid
      {...props}
      className="list-container"
      ref={ref}
      gridTemplateColumns={`repeat(auto-fill, minmax(${galleryImageMinimumWidth}px, 1fr))`}
      data-testid="image-list-container"
    >
      {props.children}
    </Grid>
  );
});

export default memo(ListContainer);
