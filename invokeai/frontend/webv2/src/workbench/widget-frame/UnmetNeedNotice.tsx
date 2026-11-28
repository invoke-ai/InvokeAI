import type { NormalizedWidgetManifest, WidgetInstanceId, WidgetManifest } from '@workbench/widgetContracts';

import { HStack, Icon, Text, VisuallyHidden } from '@chakra-ui/react';
import { Button } from '@platform/ui/Button';
import { resolveWidgetLabel } from '@workbench/widgetLabels';
import { useActiveLayoutSelector, useWorkbenchQueries } from '@workbench/WorkbenchContext';
import { useWorkbenchWidgetRegistry } from '@workbench/WorkbenchWidgetRegistryContext';
import { getUnmetNeed } from '@workbench/workspace/unmetNeeds';
import { useWorkspaceOpen } from '@workbench/workspace/useWorkspaceOpen';
import { EyeOffIcon } from 'lucide-react';
import { useCallback } from 'react';
import { useTranslation } from 'react-i18next';

/**
 * The frame's one-line notice while what the widget acts on (`needs`) is not on screen, with the gesture that shows it.
 * It sits at the foot of the body, so it shortens the body rather than moving what the user works with. Shell chrome:
 * the widget renders as usual and knows nothing of it.
 */
export const UnmetNeedNotice = ({
  instanceId,
  manifest,
}: {
  instanceId: WidgetInstanceId;
  manifest: Pick<WidgetManifest | NormalizedWidgetManifest, 'needs'>;
}) => (manifest.needs?.length ? <UnmetNeedStrip instanceId={instanceId} needs={manifest.needs} /> : null);

const UnmetNeedStrip = ({ instanceId, needs }: { instanceId: WidgetInstanceId; needs: readonly string[] }) => {
  const { t } = useTranslation();
  const queries = useWorkbenchQueries();
  const { getWidgetsForRegion } = useWorkbenchWidgetRegistry();
  const open = useWorkspaceOpen();
  const { getPresentation } = queries.getAvailability();
  // A scalar, so the frame re-renders only when the notice itself changes.
  const need = useActiveLayoutSelector((layout) => {
    const unmet = getUnmetNeed(layout, instanceId, needs, getPresentation);

    return unmet ? `${unmet.isPlacedHere ? 'hidden' : 'elsewhere'}:${unmet.typeId}` : null;
  });
  const separator = need?.indexOf(':') ?? -1;
  const placement = need === null ? null : need.slice(0, separator);
  const typeId = need === null ? null : need.slice(separator + 1);
  const widget =
    typeId === null
      ? undefined
      : (['center', 'left', 'right', 'bottom'] as const)
          .flatMap((region) => getWidgetsForRegion(region))
          .find((candidate) => candidate.manifest.id === typeId);
  const label = widget ? resolveWidgetLabel(widget.manifest, t) : (typeId ?? '');
  const text =
    placement === null
      ? ''
      : t(placement === 'hidden' ? 'widgets.needs.hidden' : 'widgets.needs.elsewhere', { widget: label });
  const show = useCallback(() => {
    if (typeId !== null) {
      open({ typeId });
    }
  }, [open, typeId]);

  return (
    <>
      {/* Mounted with the frame, so the notice is announced once, when it appears. */}
      <VisuallyHidden role="status">{text}</VisuallyHidden>
      {placement === null ? null : (
        <HStack
          bg="bg.muted"
          borderColor="border.subtle"
          borderTopWidth="1px"
          flexShrink={0}
          gap="1.5"
          h="7"
          minW="0"
          pe="1"
          ps="2"
          data-unmet-need={typeId}
        >
          <Icon as={EyeOffIcon} boxSize="3.5" color="fg.muted" flexShrink={0} />
          <Text color="fg.muted" flex="1" fontSize="sm" minW="0" truncate>
            {text}
          </Text>
          <Button flexShrink={0} size="xs" variant="ghost" onClick={show}>
            {t('widgets.needs.show', { widget: label })}
          </Button>
        </HStack>
      )}
    </>
  );
};
