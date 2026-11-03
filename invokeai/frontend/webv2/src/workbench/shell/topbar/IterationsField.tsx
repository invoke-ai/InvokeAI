import { NumberInput, VisuallyHidden } from '@chakra-ui/react';
import { MIN_BATCH_COUNT } from '@features/generation/settings';
import { Tooltip } from '@platform/ui/Tooltip';
import { getProjectWidgetValues } from '@workbench/widgetState';
import { useActiveProjectSelector, useWorkbenchCommands } from '@workbench/WorkbenchContext';
import { useCallback, useId } from 'react';
import { useTranslation } from 'react-i18next';

import { getBatchCount, isBatchCountFixed } from './useInvocationState';

/**
 * Invoke's rounded corners overlap this field, so its edge facing Invoke is hidden and a tail continues the frame
 * beneath them. The input is a replaced element, so the root carries the tail; it overlaps the input by 2px so the
 * opaque border rows join without a sub-pixel seam.
 */
const TAIL_UNDER_INVOKE_AFTER_CSS = {
  '& input': { borderInlineEndColor: 'transparent' },
  '&::after': {
    borderBlockWidth: '1px',
    content: '""',
    pointerEvents: 'none',
    position: 'absolute',
    borderColor: 'border',
    insetBlock: 0,
    insetInlineStart: 'calc(100% - 2px)',
    width: 'calc({radii.control} + 2px)',
  },
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
        css={TAIL_UNDER_INVOKE_AFTER_CSS}
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
