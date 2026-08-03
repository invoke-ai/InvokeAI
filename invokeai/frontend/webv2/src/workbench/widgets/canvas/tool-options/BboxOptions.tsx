import type { AspectRatioId } from '@features/generation/contracts';

import { HStack } from '@chakra-ui/react';
import { AspectRatioLockButton, AspectRatioSelect } from '@features/generation/components';
import { ASPECT_RATIO_MAP, deriveAspectRatioId } from '@features/generation/settings';
import { constrainBboxToRatio } from '@workbench/canvas-engine/api';
import { useCallback } from 'react';

import type { ToolOptionsComponentProps } from './ToolOptionsBar';

import { useBboxEditor } from './useBboxEditor';

const ASPECT_TRIGGER_PROPS = { minW: '7.5rem' } as const;

/**
 * Bbox tool options: aspect-ratio preset select and aspect lock, sharing the
 * generate widget's controls so both surfaces offer the same presets — the two
 * are already kept in sync by `canvasDimsSync`. Numeric frame details live in
 * the separate bbox details bar.
 */
export const BboxOptions = ({ engine }: ToolOptionsComponentProps) => {
  const { bbox, commitBbox, grid, options } = useBboxEditor(engine);
  const bboxRatio = bbox.height > 0 ? bbox.width / bbox.height : 1;
  // Derived from the live frame rather than the stored ratio, so a lock captured
  // from a hand-drawn bbox reports the preset it actually matches.
  const selectedId: AspectRatioId = options.aspectLocked ? deriveAspectRatioId(bbox.width, bbox.height) : 'Free';

  const onAspectPresetChange = useCallback(
    (id: AspectRatioId) => {
      if (id === 'Free') {
        engine.interaction.set('bboxOptions', { ...options, aspectLocked: false });
        return;
      }

      const ratio = ASPECT_RATIO_MAP[id].ratio;

      engine.interaction.set('bboxOptions', { aspectLocked: true, aspectRatio: ratio });
      // The bbox is a placed frame, so reshape it in place rather than
      // re-deriving a free size from its pixel area.
      commitBbox(constrainBboxToRatio(bbox, ratio, grid));
    },
    [bbox, commitBbox, engine, grid, options]
  );

  const onLockToggle = useCallback(() => {
    const checked = !options.aspectLocked;
    // Locking a frame that matches no preset captures its current ratio so
    // further edits preserve it.
    const aspectRatio =
      checked && bbox.height > 0 && deriveAspectRatioId(bbox.width, bbox.height) === 'Free'
        ? bboxRatio
        : options.aspectRatio > 0
          ? options.aspectRatio
          : 1;

    engine.interaction.set('bboxOptions', { aspectLocked: checked, aspectRatio });
  }, [bbox, bboxRatio, engine, options.aspectLocked, options.aspectRatio]);

  return (
    <HStack align="center" gap="1">
      <AspectRatioSelect
        fallbackRatio={bboxRatio}
        triggerProps={ASPECT_TRIGGER_PROPS}
        value={selectedId}
        onChange={onAspectPresetChange}
      />
      <AspectRatioLockButton isLocked={options.aspectLocked} onToggle={onLockToggle} />
    </HStack>
  );
};
