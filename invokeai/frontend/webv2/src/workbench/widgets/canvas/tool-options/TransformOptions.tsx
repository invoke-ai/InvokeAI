import type { NumberInput as ChakraNumberInput } from '@chakra-ui/react';
import type { LayerTransform } from '@workbench/canvas-engine/transform/transformMath';

import { HStack, NumberInput, Text } from '@chakra-ui/react';
import { Button } from '@workbench/components/ui';
import { useTransformSession } from '@workbench/widgets/canvas/engineStoreHooks';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

const RAD_TO_DEG = 180 / Math.PI;
const DEG_TO_RAD = Math.PI / 180;

/** Rounds to 2 decimals for display without trailing float noise. */
const round2 = (n: number): number => Math.round(n * 100) / 100;

/**
 * Transform tool options: numeric X / Y (document px), width/height scale
 * (percent), and rotation (degrees) of the active session, plus Apply / Cancel.
 *
 * Scale is shown as a PERCENT of the layer's native size rather than absolute
 * W/H pixels: the session carries only the transform (not the source pixel
 * dimensions), so a percent is the resolution-independent, self-contained
 * representation. Edits update the session preview through the engine (no
 * dispatch until Apply); the whole session is one undo entry. Disabled with no
 * session.
 */
export const TransformOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { t } = useTranslation();
  const session = useTransformSession(engine);
  const transform = session?.transform ?? null;

  const patch = useCallback(
    (next: Partial<LayerTransform>) => {
      if (!transform) {
        return;
      }
      engine.layers.updateTransformSession({ ...transform, ...next });
    },
    [engine, transform]
  );

  const onX = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        patch({ x: Math.round(valueAsNumber) });
      }
    },
    [patch]
  );
  const onY = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        patch({ y: Math.round(valueAsNumber) });
      }
    },
    [patch]
  );
  const onScaleX = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber) && valueAsNumber !== 0) {
        patch({ scaleX: valueAsNumber / 100 });
      }
    },
    [patch]
  );
  const onScaleY = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber) && valueAsNumber !== 0) {
        patch({ scaleY: valueAsNumber / 100 });
      }
    },
    [patch]
  );
  const onRotation = useCallback(
    ({ valueAsNumber }: ChakraNumberInput.ValueChangeDetails) => {
      if (Number.isFinite(valueAsNumber)) {
        patch({ rotation: valueAsNumber * DEG_TO_RAD });
      }
    },
    [patch]
  );

  const onApply = useCallback(() => engine.layers.applyTransform(), [engine]);
  const onCancel = useCallback(() => engine.layers.cancelTransform(), [engine]);

  const disabled = !transform;

  return (
    <HStack align="center" gap="3">
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.positionX')}
        </Text>
        <NumberInput.Root
          disabled={disabled}
          size="xs"
          value={transform ? String(Math.round(transform.x)) : ''}
          w="4.5rem"
          onValueChange={onX}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.positionX')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.positionY')}
        </Text>
        <NumberInput.Root
          disabled={disabled}
          size="xs"
          value={transform ? String(Math.round(transform.y)) : ''}
          w="4.5rem"
          onValueChange={onY}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.positionY')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.scaleWidth')}
        </Text>
        <NumberInput.Root
          disabled={disabled}
          size="xs"
          value={transform ? String(round2(transform.scaleX * 100)) : ''}
          w="4.5rem"
          onValueChange={onScaleX}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.scaleWidth')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.scaleHeight')}
        </Text>
        <NumberInput.Root
          disabled={disabled}
          size="xs"
          value={transform ? String(round2(transform.scaleY * 100)) : ''}
          w="4.5rem"
          onValueChange={onScaleY}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.scaleHeight')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <HStack align="center" gap="1.5">
        <Text color="fg.muted" fontSize="2xs">
          {t('widgets.canvas.toolOptions.rotation')}
        </Text>
        <NumberInput.Root
          disabled={disabled}
          size="xs"
          value={transform ? String(round2(transform.rotation * RAD_TO_DEG)) : ''}
          w="4.5rem"
          onValueChange={onRotation}
        >
          <NumberInput.Control />
          <NumberInput.Input aria-label={t('widgets.canvas.toolOptions.rotation')} fontSize="xs" />
        </NumberInput.Root>
      </HStack>
      <Button disabled={disabled} size="xs" variant="solid" onClick={onApply}>
        {t('widgets.canvas.toolOptions.applyTransform')}
      </Button>
      <Button disabled={disabled} size="xs" variant="ghost" onClick={onCancel}>
        {t('widgets.canvas.toolOptions.cancelTransform')}
      </Button>
    </HStack>
  );
};
