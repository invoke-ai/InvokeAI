/* eslint-disable i18next/no-literal-string */
import type { ButtonGroupProps } from '@invoke-ai/ui-library';
import { Button, ButtonGroup } from '@invoke-ai/ui-library';
import { useAppStore } from 'app/store/nanostores/store';
import { newCanvasFromImage } from 'features/imageActions/actions';
import { memo, useCallback } from 'react';
import type { ImageDTO } from 'services/api/types';

export const ImageActions = memo(({ imageDTO, ...rest }: { imageDTO: ImageDTO } & ButtonGroupProps) => {
  const { getState, dispatch } = useAppStore();

  const vary = useCallback(() => {
    newCanvasFromImage({
      imageDTO,
      type: 'raster_layer',
      withResize: true,
      getState,
      dispatch,
    });
  }, [dispatch, getState, imageDTO]);

  const useAsControl = useCallback(() => {
    newCanvasFromImage({
      imageDTO,
      type: 'control_layer',
      withResize: true,
      getState,
      dispatch,
    });
  }, [dispatch, getState, imageDTO]);

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
      <Button onClick={vary} tooltip="Vary the image using Image to Image">
        Vary
      </Button>
      <Button onClick={useAsControl} tooltip="Use this image to control a new Text to Image generation">
        Use as Control
      </Button>
      <Button onClick={edit} tooltip="Edit parts of this image with Inpainting">
        Edit
      </Button>
    </ButtonGroup>
  );
});
ImageActions.displayName = 'ImageActions';
