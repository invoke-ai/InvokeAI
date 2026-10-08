/* oxlint-disable react-perf/jsx-no-new-array-as-prop -- hint arrays are memoized from live focus and binding snapshots. */
import type { ShortcutHint, ShortcutHintSnapshot } from '@workbench/hotkeys/hintSources';
import type { ResolvedShortcutHint } from '@workbench/hotkeys/shortcutHints';
import type { WidgetViewProps } from '@workbench/widgetContracts';

import { Box, Flex, Icon, Popover, Stack, Text, usePopoverContext } from '@chakra-ui/react';
import { Button } from '@platform/ui';
import { useCurrentWorkbenchFocusTarget } from '@workbench/focusRegions';
import { firstPartyHotkeyCatalog } from '@workbench/hotkeys/catalog';
import { useExtensionHotkeyDefinitions } from '@workbench/hotkeys/extensionHotkeys';
import { useShortcutHintSource, useShortcutFocusElement } from '@workbench/hotkeys/hintSources';
import { ShortcutKeycaps } from '@workbench/hotkeys/keyGlyphs';
import { isEditableHotkeyTarget } from '@workbench/hotkeys/keys';
import { applyCustomHotkeys } from '@workbench/hotkeys/resolve';
import {
  getVisibleShortcutHintCount,
  isShortcutGuideSessionKey,
  resolveShortcutHints,
} from '@workbench/hotkeys/shortcutHints';
import { getHotkeyTargetWidget, resolveHotkeyTarget } from '@workbench/hotkeys/targetWidget';
import { openWorkbenchSettings } from '@workbench/settings/settingsDialogStore';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import { useCompactWidgetCapacity } from '@workbench/widget-frame/compactWidgetCapacity';
import { areWidgetPlacementProjectsEqual, getWidgetPlacementProject } from '@workbench/widgetPlacementMeta';
import { useActiveProjectSelector } from '@workbench/WorkbenchContext';
import { KeyboardIcon } from 'lucide-react';
import { type KeyboardEvent, useCallback, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';

const GLOBAL_HINTS: ShortcutHintSnapshot = {
  hints: [
    { commandId: 'app.invoke', labelKey: 'workbench.shortcuts.actions.invoke' },
    { commandId: 'app.cancelQueueItem', labelKey: 'workbench.shortcuts.actions.cancelGeneration' },
    { commandId: 'app.openCommandPalette', labelKey: 'workbench.shortcuts.actions.commandPalette' },
  ],
  titleKey: 'workbench.shortcuts.global',
};
const SURFACE_HINTS: Record<string, ShortcutHintSnapshot> = {
  gallery: {
    titleKey: 'widgets.labels.gallery',
    hints: [
      { commandId: 'gallery.galleryNavRight', labelKey: 'workbench.shortcuts.actions.nextImage' },
      { commandId: 'gallery.extendSelectionRight', labelKey: 'workbench.shortcuts.actions.extendSelection' },
      { commandId: 'gallery.selectAllOnPage', labelKey: 'workbench.shortcuts.actions.selectPage' },
    ],
  },
  workflow: {
    titleKey: 'widgets.labels.workflow',
    hints: [
      { commandId: 'workflows.addNode', labelKey: 'workbench.shortcuts.actions.addNode' },
      { commandId: 'workflows.selectAll', labelKey: 'workbench.shortcuts.actions.selectNodes' },
    ],
  },
  preview: {
    titleKey: 'widgets.labels.preview',
    hints: [
      { commandId: 'viewer.nextComparisonMode', labelKey: 'workbench.shortcuts.actions.comparison' },
      { commandId: 'viewer.swapImages', labelKey: 'workbench.shortcuts.actions.swapImages' },
    ],
  },
  generate: {
    titleKey: 'widgets.labels.generate',
    hints: [],
  },
};

export const ShortcutGuide = ({ presentation = 'compact' }: Pick<WidgetViewProps, 'presentation'> = {}) => {
  const project = useActiveProjectSelector(getWidgetPlacementProject, areWidgetPlacementProjectsEqual);
  return <ConnectedShortcutGuide key={project.projectId} presentation={presentation} project={project} />;
};

const ConnectedShortcutGuide = ({
  project,
  presentation,
}: {
  project: ReturnType<typeof getWidgetPlacementProject>;
  presentation: WidgetViewProps['presentation'];
}) => {
  const focusTarget = useCurrentWorkbenchFocusTarget();
  const element = useShortcutFocusElement();
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const extensionHotkeys = useExtensionHotkeyDefinitions();
  const target = useMemo(
    () => resolveHotkeyTarget({ focusTarget, project, targetWidget: getHotkeyTargetWidget(element) }),
    [element, focusTarget, project]
  );
  const isProperties = target.activeWidgetTypeId === 'layers';
  const canvasInstance = isProperties
    ? (Object.values(project.widgetInstances).find((instance) => instance.typeId === 'canvas')?.id ?? null)
    : target.activeWidgetTypeId === 'canvas'
      ? target.activeInstanceId
      : null;
  const canvasHints = useShortcutHintSource(project.projectId ?? '', canvasInstance);
  const hotkeys = useMemo(
    () => [...firstPartyHotkeyCatalog, ...extensionHotkeys].map((hotkey) => applyCustomHotkeys(hotkey, customHotkeys)),
    [customHotkeys, extensionHotkeys]
  );
  // Without a mounted canvas source the guide lists only global hints, so it must not title itself as on-canvas.
  const onCanvas =
    canvasHints !== null &&
    (isProperties || (target.activeWidgetTypeId === 'canvas' && isEditableHotkeyTarget(element)));
  const snapshot = canvasHints ?? SURFACE_HINTS[target.activeWidgetTypeId ?? ''] ?? GLOBAL_HINTS;
  const resolved = useMemo(() => {
    const hints: readonly ShortcutHint[] = canvasHints
      ? snapshot.hints.filter((hint) => !onCanvas || !('commandId' in hint))
      : [...snapshot.hints, ...(snapshot === GLOBAL_HINTS ? [] : GLOBAL_HINTS.hints)];
    return resolveShortcutHints(
      hints,
      hotkeys,
      // The status-bar host unmounts this preserve-focus widget while a modal is present.
      { ...target, isModalPresent: false, projectId: project.projectId ?? '' },
      element
    );
  }, [canvasHints, element, hotkeys, onCanvas, project.projectId, snapshot, target]);
  return (
    <ShortcutGuideContent
      hints={resolved}
      onCanvas={onCanvas}
      returnFocus={element instanceof HTMLElement ? element : null}
      presentation={presentation}
      titleKey={snapshot.titleKey}
    />
  );
};

const Hint = ({ hint, compact = false }: { hint: ResolvedShortcutHint; compact?: boolean }) => {
  const { t } = useTranslation();
  return (
    <Flex align="center" flexShrink={0} gap="1" whiteSpace={compact ? 'nowrap' : undefined}>
      {hint.parts.length > 0 ? <ShortcutKeycaps parts={hint.parts} /> : null}
      {hint.pointerKey ? (
        <Text as="span" fontSize="xs">
          {t(hint.pointerKey)}
        </Text>
      ) : null}
      <Text as="span" fontSize="xs">
        {hint.labelKey ? t(hint.labelKey) : hint.title}
      </Text>
    </Flex>
  );
};

/** Measured keycaps keep whole hints in one line; the status-bar host owns movement and its popover. */
export const ShortcutGuideContent = ({
  hints,
  onCanvas = false,
  presentation = 'compact',
  returnFocus = null,
  titleKey,
}: {
  hints: readonly ResolvedShortcutHint[];
  onCanvas?: boolean;
  presentation?: WidgetViewProps['presentation'];
  returnFocus?: HTMLElement | null;
  titleKey: string;
}) => {
  const { t } = useTranslation();
  const [visibleCount, setVisibleCount] = useState(0);
  const [capacityWidth, setCapacityWidth] = useState<number>();
  const [showFallbackLabel, setShowFallbackLabel] = useState(true);
  // The bottom-only manifest guarantees both presentations are hosted by the same Popover.Root.
  const popover = usePopoverContext();
  const capacity = useCompactWidgetCapacity();
  const title = onCanvas ? t('workbench.shortcuts.onCanvas', { tool: t(titleKey) }) : t(titleKey);
  const measurementRoot = useRef<HTMLDivElement>(null);
  // Keycap/text changes need a pre-paint measurement; only this widget's own root is observed.
  useLayoutEffect(() => {
    const root = measurementRoot.current;
    if (!root) {
      return;
    }
    const measure = () => {
      const row = root.querySelector('[data-hint-measurements]');
      const probe = root.querySelector('[data-hint-capacity]');
      const [heading, ...items] = row ? [...row.children] : [];
      if (!heading || !probe) {
        return;
      }
      const available = Math.max(24, Math.min(capacity, probe.getBoundingClientRect().width));
      const widths = items.map((item) => item.getBoundingClientRect().width + 8);
      setCapacityWidth(available - 12);
      setVisibleCount(getVisibleShortcutHintCount(available, heading.getBoundingClientRect().width + 12, widths));
      const fallback = root.querySelector('[data-hint-fallback]');
      setShowFallbackLabel((fallback?.getBoundingClientRect().width ?? 0) + 12 <= available);
    };
    const observer = new ResizeObserver(measure);
    observer.observe(root);
    measure();
    return () => observer.disconnect();
  }, [capacity, hints, title]);
  const stopKeyPropagation = useCallback((event: KeyboardEvent) => {
    if (isShortcutGuideSessionKey(event)) {
      event.stopPropagation();
    }
  }, []);
  const openSettings = useCallback(() => {
    popover.setOpen(false);
    openWorkbenchSettings('hotkeys', returnFocus ?? undefined);
  }, [popover, returnFocus]);
  if (presentation !== 'compact') {
    return (
      <Box data-shortcut-guide data-workbench-focus-preserve p="3" onKeyDown={stopKeyPropagation}>
        <Popover.Title fontSize="md" fontWeight="semibold">
          {title}
        </Popover.Title>
        <Stack
          as="ul"
          gap="3"
          listStyleType="none"
          m="0"
          maxH="min(20rem, var(--available-height))"
          overflowY="auto"
          p="0"
          py="3"
        >
          {hints.length === 0 ? (
            <Text as="li" color="fg.muted" fontSize="sm">
              {t('workbench.shortcuts.empty')}
            </Text>
          ) : null}
          {hints.map((hint) => (
            <Box as="li" key={hint.id}>
              <Hint hint={hint} />
            </Box>
          ))}
        </Stack>
        <Button size="sm" variant="outline" onClick={openSettings}>
          {t('workbench.shortcuts.settings')}
        </Button>
      </Box>
    );
  }
  return (
    <Box
      ref={measurementRoot}
      data-shortcut-guide
      data-workbench-focus-preserve
      h="full"
      maxW={capacityWidth ?? 'min(40vw, 34rem)'}
      minW="0"
      overflow="hidden"
      position="relative"
    >
      <Box aria-hidden="true" data-hint-capacity position="absolute" w="min(40vw, 34rem)" />
      <Flex
        aria-hidden="true"
        data-hint-fallback
        align="center"
        gap="1"
        position="absolute"
        visibility="hidden"
        w="max-content"
      >
        <Icon as={KeyboardIcon} boxSize="3" />
        <Text fontSize="xs">{t('workbench.shortcuts.title')}</Text>
      </Flex>
      <Flex
        aria-hidden="true"
        data-hint-measurements
        gap="0"
        pointerEvents="none"
        position="absolute"
        visibility="hidden"
        w="max-content"
      >
        <Text fontSize="xs">{title}</Text>
        {hints.map((hint) => (
          <Hint key={hint.id} compact hint={hint} />
        ))}
      </Flex>
      <Flex align="center" gap="2" h="full" justify="flex-start" minW="0">
        {visibleCount > 0 ? (
          <Text color="fg.muted" flexShrink={0} fontSize="xs">
            {title}
          </Text>
        ) : (
          <Flex align="center" gap="1">
            <Icon as={KeyboardIcon} boxSize="3" />
            {showFallbackLabel ? <Text fontSize="xs">{t('workbench.shortcuts.title')}</Text> : null}
          </Flex>
        )}
        {hints.slice(0, visibleCount).map((hint) => (
          <Hint key={hint.id} compact hint={hint} />
        ))}
      </Flex>
    </Box>
  );
};
