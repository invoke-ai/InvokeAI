import type { ButtonGroupProps } from '@invoke-ai/ui-library';
import { Button, ButtonGroup } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/storeHooks';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { memo, useCallback } from 'react';
import type { ImageDTO } from 'services/api/types';

export const ImageActions = memo(({ imageDTO, ...rest }: { imageDTO: ImageDTO } & ButtonGroupProps) => {
  const { getState, dispatch } = useAppStore();

  const edit = useCallback(() => {
    newCanvasFromImage({
      imageDTO,
      type: 'raster_layer',
      withInpaintMask: true,
      getState,
      dispatch,
    });
  }, [dispatch, getState, imageDTO]);
  return (
    <ButtonGroup isAttached={false} size="sm" {...rest}>
      <Button onClick={edit} tooltip="Edit parts of this image with Inpainting">
        Edit
      </Button>
    </ButtonGroup>
  );
});
ImageActions.displayName = 'ImageActions';
