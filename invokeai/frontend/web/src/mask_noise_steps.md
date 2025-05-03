# Adding Image Noise Slider to InpaintMask Layer

This document outlines the steps required to add an "Image Noise" slider feature to the InpaintMask layer type in the Invoke AI frontend.

## Overview

The feature allows users to add and adjust an image noise level to an InpaintMask layer, similar to how Regional Guidance layers have the ability to add prompts. The implementation includes:

1. Extending the InpaintMask data model
2. Creating UI components for adding and adjusting noise
3. Adding Redux actions to handle state changes
4. Adding a delete button to remove the noise setting
5. Implementing conditional rendering of the add button based on current state

## Implementation Steps

### 1. Extend the InpaintMask Data Model

Update the `CanvasInpaintMaskState` type in `types.ts` to include a `noiseLevel` property:

```typescript
const zCanvasInpaintMaskState = zCanvasEntityBase.extend({
  type: z.literal('inpaint_mask'),
  position: zCoordinate,
  fill: zFill,
  opacity: zOpacity,
  objects: z.array(zCanvasObjectState),
  noiseLevel: z.number().gte(0).lte(1).nullable().default(null),
});
export type CanvasInpaintMaskState = z.infer<typeof zCanvasInpaintMaskState>;
```

### 2. Update the Initial State Function

Modify the `getInpaintMaskState` function in `util.ts` to include the new `noiseLevel` property:

```typescript
export const getInpaintMaskState = (
  id: string,
  overrides?: Partial<CanvasInpaintMaskState>
): CanvasInpaintMaskState => {
  const entityState: CanvasInpaintMaskState = {
    id,
    name: null,
    type: 'inpaint_mask',
    isEnabled: true,
    isLocked: false,
    objects: [],
    opacity: 1,
    position: { x: 0, y: 0 },
    fill: {
      style: 'diagonal',
      color: getInpaintMaskFillColor(),
    },
    noiseLevel: null,
  };
  merge(entityState, overrides);
  return entityState;
};
```

### 3. Add Redux Actions

Add new actions to the `canvasSlice.ts` file to handle adding, changing, and deleting noise:

```typescript
inpaintMaskNoiseAdded: (state, action: PayloadAction<EntityIdentifierPayload<void, 'inpaint_mask'>>) => {
  const { entityIdentifier } = action.payload;
  const entity = selectEntity(state, entityIdentifier);
  if (entity && entity.type === 'inpaint_mask') {
    entity.noiseLevel = 0.5; // Default noise level
  }
},
inpaintMaskNoiseChanged: (state, action: PayloadAction<EntityIdentifierPayload<{ noiseLevel: number }, 'inpaint_mask'>>) => {
  const { entityIdentifier, noiseLevel } = action.payload;
  const entity = selectEntity(state, entityIdentifier);
  if (entity && entity.type === 'inpaint_mask') {
    entity.noiseLevel = noiseLevel;
  }
},
inpaintMaskNoiseDeleted: (state, action: PayloadAction<EntityIdentifierPayload<void, 'inpaint_mask'>>) => {
  const { entityIdentifier } = action.payload;
  const entity = selectEntity(state, entityIdentifier);
  if (entity && entity.type === 'inpaint_mask') {
    entity.noiseLevel = null;
  }
},
```

Make sure to export these actions from the canvasSlice:

```typescript
export const {
  // ... other exports
  inpaintMaskNoiseAdded,
  inpaintMaskNoiseChanged,
  inpaintMaskNoiseDeleted,
} = canvasSlice.actions;
```

### 4. Create a Hook for Adding Noise

Add a new hook in `addLayerHooks.ts` to handle adding noise to an InpaintMask:

```typescript
export const useAddInpaintMaskNoise = (entityIdentifier: CanvasEntityIdentifier<'inpaint_mask'>) => {
  const dispatch = useAppDispatch();
  const func = useCallback(() => {
    dispatch(inpaintMaskNoiseAdded({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  return func;
};
```

### 5. Create UI Components

#### 5.1. Create a Button Component for Adding Noise

Create a new file `InpaintMaskAddNoiseButton.tsx` for the button that adds noise:

```typescript
import { Button, Flex } from '@invoke-ai/ui-library';
import { useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { useAddInpaintMaskNoise } from 'features/controlLayers/hooks/addLayerHooks';
import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiPlusBold } from 'react-icons/pi';

export const InpaintMaskAddNoiseButton = () => {
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const { t } = useTranslation();
  const addInpaintMaskNoise = useAddInpaintMaskNoise(entityIdentifier);

  return (
    <Flex w="full" p={2} justifyContent="center">
      <Button
        size="sm"
        variant="ghost"
        leftIcon={<PiPlusBold />}
        onClick={addInpaintMaskNoise}
      >
        {t('controlLayers.imageNoise')}
      </Button>
    </Flex>
  );
};
```

#### 5.2. Create a Delete Button Component

Create a new file `InpaintMaskDeleteNoiseButton.tsx` for the X button that deletes noise:

```typescript
import type { IconButtonProps } from '@invoke-ai/ui-library';
import { IconButton } from '@invoke-ai/ui-library';
import { memo } from 'react';
import { useTranslation } from 'react-i18next';
import { PiXBold } from 'react-icons/pi';

type Props = Omit<IconButtonProps, 'aria-label'> & {
  onDelete: () => void;
};

export const InpaintMaskDeleteNoiseButton = memo(({ onDelete, ...rest }: Props) => {
  const { t } = useTranslation();
  return (
    <IconButton
      tooltip={t('common.delete')}
      variant="link"
      aria-label={t('common.delete')}
      icon={<PiXBold />}
      onClick={onDelete}
      flexGrow={0}
      size="sm"
      p={0}
      colorScheme="error"
      {...rest}
    />
  );
});

InpaintMaskDeleteNoiseButton.displayName = 'InpaintMaskDeleteNoiseButton';
```

#### 5.3. Create a Slider Component with Delete Button

Create a new file `InpaintMaskNoiseSlider.tsx` for adjusting the noise level and including the delete button:

```typescript
import { Flex, Slider, SliderFilledTrack, SliderThumb, SliderTrack, Text } from '@invoke-ai/ui-library';
import { createSelector } from '@reduxjs/toolkit';
import { useAppDispatch, useAppSelector } from 'app/store/storeHooks';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { inpaintMaskNoiseChanged, inpaintMaskNoiseDeleted } from 'features/controlLayers/store/canvasSlice';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import { memo, useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { InpaintMaskDeleteNoiseButton } from './InpaintMaskDeleteNoiseButton';

export const InpaintMaskNoiseSlider = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const { t } = useTranslation();
  const dispatch = useAppDispatch();

  const selectNoiseLevel = useMemo(
    () =>
      createSelector(
        selectCanvasSlice,
        (canvas) => selectEntityOrThrow(canvas, entityIdentifier, 'InpaintMaskNoiseSlider').noiseLevel
      ),
    [entityIdentifier]
  );
  const noiseLevel = useAppSelector(selectNoiseLevel);

  const handleNoiseChange = useCallback(
    (value: number) => {
      dispatch(inpaintMaskNoiseChanged({ entityIdentifier, noiseLevel: value }));
    },
    [dispatch, entityIdentifier]
  );

  const onDeleteNoise = useCallback(() => {
    dispatch(inpaintMaskNoiseDeleted({ entityIdentifier }));
  }, [dispatch, entityIdentifier]);

  if (noiseLevel === null) {
    return null;
  }

  return (
    <Flex direction="column" gap={1} w="full" px={2} pb={2}>
      <Flex justifyContent="space-between" w="full" alignItems="center">
        <Text fontSize="sm">{t('controlLayers.imageNoise')}</Text>
        <Flex alignItems="center" gap={1}>
          <Text fontSize="sm">{Math.round(noiseLevel * 100)}%</Text>
          <InpaintMaskDeleteNoiseButton onDelete={onDeleteNoise} />
        </Flex>
      </Flex>
      <Slider
        aria-label={t('controlLayers.imageNoise')}
        value={noiseLevel}
        min={0}
        max={1}
        step={0.01}
        onChange={handleNoiseChange}
      >
        <SliderTrack>
          <SliderFilledTrack />
        </SliderTrack>
        <SliderThumb />
      </Slider>
    </Flex>
  );
});

InpaintMaskNoiseSlider.displayName = 'InpaintMaskNoiseSlider';
```

### 6. Create a Settings Component for Conditional Rendering

Create a new file `InpaintMaskSettings.tsx` to handle conditional rendering of components based on state:

```typescript
import { createMemoizedSelector } from 'app/store/createMemoizedSelector';
import { useAppSelector } from 'app/store/storeHooks';
import { CanvasEntitySettingsWrapper } from 'features/controlLayers/components/common/CanvasEntitySettingsWrapper';
import { InpaintMaskAddNoiseButton } from 'features/controlLayers/components/InpaintMask/InpaintMaskAddNoiseButton';
import { InpaintMaskNoiseSlider } from 'features/controlLayers/components/InpaintMask/InpaintMaskNoiseSlider';
import { useEntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import { selectCanvasSlice, selectEntityOrThrow } from 'features/controlLayers/store/selectors';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

const buildSelectFlags = (entityIdentifier: CanvasEntityIdentifier<'inpaint_mask'>) =>
  createMemoizedSelector(selectCanvasSlice, (canvas) => {
    const entity = selectEntityOrThrow(canvas, entityIdentifier, 'InpaintMaskSettings');
    return {
      hasNoiseLevel: entity.noiseLevel !== null,
    };
  });

export const InpaintMaskSettings = memo(() => {
  const entityIdentifier = useEntityIdentifierContext('inpaint_mask');
  const selectFlags = useMemo(() => buildSelectFlags(entityIdentifier), [entityIdentifier]);
  const flags = useAppSelector(selectFlags);

  return (
    <CanvasEntitySettingsWrapper>
      {!flags.hasNoiseLevel && <InpaintMaskAddNoiseButton />}
      {flags.hasNoiseLevel && <InpaintMaskNoiseSlider />}
    </CanvasEntitySettingsWrapper>
  );
});

InpaintMaskSettings.displayName = 'InpaintMaskSettings';
```

### 7. Update the InpaintMask Component

Modify the `InpaintMask.tsx` file to use the new settings component:

```typescript
import { Spacer } from '@invoke-ai/ui-library';
import { CanvasEntityContainer } from 'features/controlLayers/components/CanvasEntityList/CanvasEntityContainer';
import { CanvasEntityHeader } from 'features/controlLayers/components/common/CanvasEntityHeader';
import { CanvasEntityHeaderCommonActions } from 'features/controlLayers/components/common/CanvasEntityHeaderCommonActions';
import { CanvasEntityPreviewImage } from 'features/controlLayers/components/common/CanvasEntityPreviewImage';
import { CanvasEntityEditableTitle } from 'features/controlLayers/components/common/CanvasEntityTitleEdit';
import { InpaintMaskSettings } from 'features/controlLayers/components/InpaintMask/InpaintMaskSettings';
import { CanvasEntityStateGate } from 'features/controlLayers/contexts/CanvasEntityStateGate';
import { InpaintMaskAdapterGate } from 'features/controlLayers/contexts/EntityAdapterContext';
import { EntityIdentifierContext } from 'features/controlLayers/contexts/EntityIdentifierContext';
import type { CanvasEntityIdentifier } from 'features/controlLayers/store/types';
import { memo, useMemo } from 'react';

type Props = {
  id: string;
};

export const InpaintMask = memo(({ id }: Props) => {
  const entityIdentifier = useMemo<CanvasEntityIdentifier<'inpaint_mask'>>(() => ({ id, type: 'inpaint_mask' }), [id]);

  return (
    <EntityIdentifierContext.Provider value={entityIdentifier}>
      <InpaintMaskAdapterGate>
        <CanvasEntityStateGate entityIdentifier={entityIdentifier}>
          <CanvasEntityContainer>
            <CanvasEntityHeader>
              <CanvasEntityPreviewImage />
              <CanvasEntityEditableTitle />
              <Spacer />
              <CanvasEntityHeaderCommonActions />
            </CanvasEntityHeader>
            <InpaintMaskSettings />
          </CanvasEntityContainer>
        </CanvasEntityStateGate>
      </InpaintMaskAdapterGate>
    </EntityIdentifierContext.Provider>
  );
});

InpaintMask.displayName = 'InpaintMask';
```

## Additional Considerations

1. Add translations for the "Image Noise" text in the translation files
2. Implement the backend functionality to process the noise level during image generation
3. Add tests for the new functionality
4. Update documentation to reflect the new feature

## Conclusion

This implementation adds an "Image Noise" slider to the InpaintMask layer, following a similar pattern to how Regional Guidance layers handle prompts. The feature allows users to add, adjust, and delete noise levels for their inpainting masks, enhancing the functionality of the application. The implementation includes conditional rendering of UI components based on the current state, ensuring that the add button is only shown when no noise level is set, and the slider with delete button is shown when a noise level exists.
