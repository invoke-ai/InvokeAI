import { NumberInput, VisuallyHidden } from '@chakra-ui/react';
import { MIN_BATCH_COUNT } from '@features/generation/settings';
import { Tooltip } from '@platform/ui/Tooltip';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useId } from 'react';
import { useTranslation } from 'react-i18next';

import { getBatchCount, isBatchCountFixed } from './useInvocationState';

/**
 * Invoke's rounded corners overlap this field. The input, the field's visible frame, runs one control radius past the
 * root beneath them and has no facing edge, so its top and bottom borders reach Invoke's straight edges in every state
 * (hover, focus, invalid) in the color the input itself draws. A zero width rather than a transparent color, because
 * state colors would repaint that edge inside an outline Invoke. The root keeps its width and the steppers stay
 * anchored to it, so layout is unchanged.
 */
const RUN_UNDER_INVOKE_CSS = {
  '& input': { borderInlineEndWidth: '0', width: 'calc(100% + {radii.control})' },
} as const;

export const IterationsField = () => {
  const { t } = useTranslation();
  const { batchCount, isFixed, sourceId } = useActiveProjectSelector(
    (project) => {
      const sourceId = project.invocation.sourceId;
      const typeId = sourceId === 'upscale' || sourceId === 'video' || sourceId === 'workflow' ? sourceId : 'generate';
      // The same resolution the patch command writes through, so reader and writer agree on the instance.
      const values = getProjectWidgetValues(project, typeId);

      return {
        batchCount: getBatchCount(sourceId, values),
        isFixed: isBatchCountFixed(sourceId, values),
        sourceId,
      };
    },
    (left, right) =>
      left.batchCount === right.batchCount && left.isFixed === right.isFixed && left.sourceId === right.sourceId
  );
  const { generation, widgets } = useWorkbenchCommands();
  const fixedReasonId = useId();
  const handleValueChange = useCallback(
    ({ valueAsNumber }: { valueAsNumber: number }) => {
      if (!Number.isFinite(valueAsNumber)) {
        return;
      }

      if (sourceId === 'upscale' || sourceId === 'video' || sourceId === 'workflow') {
        widgets.patchValues(sourceId, { batchCount: valueAsNumber });
      } else {
        generation.setBatchCount(valueAsNumber);
      }
    },
    [generation, sourceId, widgets]
  );

  return (
    <Tooltip
      content={isFixed ? t('topbar.iterations.fixedModelOnly') : t('topbar.iterations.tooltip', { count: batchCount })}
      showArrow
    >
      {/* Read-only rather than disabled while the count is fixed: the field keeps its focus and its tooltip, which
          say why it cannot change. */}
      <NumberInput.Root
        allowMouseWheel
        css={RUN_UNDER_INVOKE_CSS}
        cursor={isFixed ? 'not-allowed' : undefined}
        flexShrink={0}
        opacity={isFixed ? 0.55 : undefined}
        readOnly={isFixed}
        min={MIN_BATCH_COUNT}
        rounded="control"
        size="lg"
        value={String(batchCount)}
        w="14"
        onValueChange={handleValueChange}
      >
        <NumberInput.Control />
        {/* The visible surface is the inner control, so it inherits the root's
            radius — that is the corner an attached `Group` adjusts. */}
        <NumberInput.Input
          aria-describedby={isFixed ? fixedReasonId : undefined}
          aria-disabled={isFixed || undefined}
          aria-label={t('topbar.iterations.label')}
          paddingStart="2"
          rounded="inherit"
        />
        {isFixed ? <VisuallyHidden id={fixedReasonId}>{t('topbar.iterations.fixedModelOnly')}</VisuallyHidden> : null}
      </NumberInput.Root>
    </Tooltip>
  );
};
