import type {
  CanvasCoreStoreCapability,
  CanvasToolCapability,
  ShapeToolKind,
  ToolId,
} from '@workbench/canvas-engine/api';
import type { HotkeyDefinition } from '@workbench/hotkeys/types';

import { Box, Stack, Text } from '@chakra-ui/react';
import { Toolbar, ToolbarButton } from '@platform/ui/Toolbar';
import { GradientIcon } from '@platform/ui/VendoredIcon';
import { useExtensionHotkeyDefinitions } from '@workbench/hotkeys/extensionHotkeys';
import { formatHotkeyLabel } from '@workbench/hotkeys/keys';
import { applyCustomHotkeys } from '@workbench/hotkeys/resolve';
import { useWorkbenchPreferenceSelector } from '@workbench/settings/store';
import {
  BrushIcon,
  CircleIcon,
  EraserIcon,
  HandIcon,
  LassoIcon,
  MousePointer2Icon,
  PaintBucketIcon,
  PenLineIcon,
  PentagonIcon,
  ScalingIcon,
  ScanIcon,
  SquareDashedMousePointerIcon,
  SquareIcon,
  StarIcon,
  TriangleIcon,
  TypeIcon,
} from 'lucide-react';
import { useCallback, useEffect, useMemo, type ElementType, type ReactNode } from 'react';
import { useTranslation } from 'react-i18next';

import { useCanvasActiveTool, useShapeOptions } from './engineStoreHooks';
import {
  familyMemberOf,
  recordFamilyTool,
  useRememberedFamilyTool,
  type ToolFamily,
  type ToolFamilyMember,
} from './toolFamilyStore';
import { ToolFamilyButton, type ToolFlyoutItem } from './ToolStripFlyout';

type ToolStripEngine = CanvasCoreStoreCapability & { readonly tools: CanvasToolCapability };

interface ToolStripButtonProps {
  engine: ToolStripEngine;
  /** One help line under the title, for a tool whose key is not a plain binding (the Frame tool's C). */
  description?: string;
  /** The tool's first effective binding, shown with its name; none while unbound. */
  hotkey?: string;
  icon: ElementType;
  label: string;
  toolId: ToolId;
}

/** A tool's tooltip: its name with the key that selects it, and an optional help line. */
const useToolTooltip = (label: string, hotkey: string | undefined, description?: string): ReactNode => {
  const { t } = useTranslation();
  const title = hotkey ? t('widgets.canvas.tools.withHotkey', { hotkey: formatHotkeyLabel(hotkey), label }) : label;
  return useMemo(
    () =>
      description ? (
        <Stack gap="0.5" maxW="16rem">
          <Text>{title}</Text>
          <Text color="fg.muted" fontSize="sm">
            {description}
          </Text>
        </Stack>
      ) : (
        title
      ),
    [description, title]
  );
};

/** One sticky tool button: active state comes from core stores; click drives the tool capability. */
const ToolStripButton = ({ description, engine, hotkey, icon, label, toolId }: ToolStripButtonProps) => {
  const activeTool = useCanvasActiveTool(engine);
  const onClick = useCallback(() => engine.tools.setTool(toolId), [engine, toolId]);
  const tooltip = useToolTooltip(label, hotkey, description);

  return (
    <ToolbarButton icon={icon} isActive={activeTool === toolId} label={label} tooltip={tooltip} onClick={onClick} />
  );
};

const SHAPE_KIND_ICONS: Record<ShapeToolKind, React.ElementType> = {
  ellipse: CircleIcon,
  freehand: PenLineIcon,
  polygon: PentagonIcon,
  rect: SquareIcon,
  star: StarIcon,
  triangle: TriangleIcon,
};
const SHAPE_KIND_ORDER: readonly ShapeToolKind[] = ['rect', 'ellipse', 'triangle', 'star', 'polygon', 'freehand'];
const SHAPE_KIND_LABEL_KEYS: Record<ShapeToolKind, string> = {
  ellipse: 'widgets.canvas.toolOptions.shapeEllipse',
  freehand: 'widgets.canvas.toolOptions.shapeFreehand',
  polygon: 'widgets.canvas.toolOptions.shapePolygon',
  rect: 'widgets.canvas.toolOptions.shapeRect',
  star: 'widgets.canvas.toolOptions.shapeStar',
  triangle: 'widgets.canvas.toolOptions.shapeTriangle',
};

/** The Shape slot: one tool id, the flyout switching which kind the next drag draws. */
const ShapeFamilyButton = ({ engine, hotkey }: { engine: ToolStripEngine; hotkey: string | undefined }) => {
  const { t } = useTranslation();
  const activeTool = useCanvasActiveTool(engine);
  const options = useShapeOptions(engine);
  const items: ToolFlyoutItem[] = useMemo(
    () =>
      SHAPE_KIND_ORDER.map((kind) => ({
        // One key selects the Shape tool and cycles its kinds; each entry shows it.
        hotkeys: hotkey ? [hotkey] : undefined,
        icon: SHAPE_KIND_ICONS[kind],
        id: kind,
        label: t(SHAPE_KIND_LABEL_KEYS[kind]),
      })),
    [hotkey, t]
  );
  const onActivate = useCallback(() => engine.tools.setTool('shape'), [engine]);
  const onSelectSubtool = useCallback(
    (id: string) => {
      engine.interaction.set('shapeOptions', { ...options, kind: id as ShapeToolKind });
      engine.tools.setTool('shape');
    },
    [engine, options]
  );
  return (
    <ToolFamilyButton
      currentId={options.kind}
      icon={SHAPE_KIND_ICONS[options.kind]}
      isActive={activeTool === 'shape'}
      items={items}
      label={t('widgets.canvas.tools.shape')}
      onActivate={onActivate}
      onSelectSubtool={onSelectSubtool}
    />
  );
};

/**
 * A tool's effective bindings, remaps included, from the hotkeys the canvas widget registered; none outside a
 * workbench or while unbound.
 */
const toolHotkeys = (
  definitions: readonly HotkeyDefinition[],
  customHotkeys: Record<string, string[]>,
  hotkeyId: string
): string[] | undefined => {
  const definition = definitions.find((hotkey) => hotkey.id === hotkeyId);
  return definition ? applyCustomHotkeys(definition, customHotkeys).keys : undefined;
};

/**
 * A family slot's state: the active tool when it belongs to the family, else the remembered member. A switch into
 * the family by any route (strip, hotkey, a restored session) becomes its memory.
 */
const useFamilySlot = <Family extends ToolFamily>(engine: ToolStripEngine, family: Family) => {
  const active = familyMemberOf(family, useCanvasActiveTool(engine));
  const remembered = useRememberedFamilyTool(family);
  useEffect(() => {
    if (active !== null) {
      recordFamilyTool(family, active);
    }
  }, [active, family]);
  const current = active ?? remembered;
  const onActivate = useCallback(() => engine.tools.setTool(current), [current, engine]);
  const onSelectSubtool = useCallback(
    (id: string) => {
      const member = familyMemberOf(family, id);
      if (member !== null) {
        recordFamilyTool(family, member);
        engine.tools.setTool(member);
      }
    },
    [engine, family]
  );
  return { current, isActive: active !== null, onActivate, onSelectSubtool };
};

/** The Select slot: marquee and lasso as one family; the slot stands for the last one used. */
const SelectFamilyButton = ({ engine }: { engine: ToolStripEngine }) => {
  const { t } = useTranslation();
  const { current, isActive, onActivate, onSelectSubtool } = useFamilySlot(engine, 'select');
  const hotkeyDefinitions = useExtensionHotkeyDefinitions();
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const items: ToolFlyoutItem[] = useMemo(
    () => [
      {
        description: t('widgets.canvas.tools.marqueeHelp'),
        icon: SquareDashedMousePointerIcon,
        id: 'marquee',
        label: t('widgets.canvas.tools.marquee'),
        hotkeys: toolHotkeys(hotkeyDefinitions, customHotkeys, 'canvas.tool.marquee'),
      },
      {
        description: t('widgets.canvas.tools.lassoHelp'),
        icon: LassoIcon,
        id: 'lasso',
        label: t('widgets.canvas.tools.lasso'),
        hotkeys: toolHotkeys(hotkeyDefinitions, customHotkeys, 'canvas.tool.lasso'),
      },
    ],
    [customHotkeys, hotkeyDefinitions, t]
  );
  return (
    <ToolFamilyButton
      currentId={current}
      icon={current === 'lasso' ? LassoIcon : SquareDashedMousePointerIcon}
      isActive={isActive}
      items={items}
      label={t(current === 'lasso' ? 'widgets.canvas.tools.lasso' : 'widgets.canvas.tools.marquee')}
      onActivate={onActivate}
      onSelectSubtool={onSelectSubtool}
    />
  );
};

const FILL_FAMILY_ICONS: Record<ToolFamilyMember<'fill'>, ElementType> = {
  gradient: GradientIcon,
  paintBucket: PaintBucketIcon,
};

/** The Fill slot: Gradient and Paint Bucket as one family, which G selects and toggles between. */
const FillFamilyButton = ({ engine }: { engine: ToolStripEngine }) => {
  const { t } = useTranslation();
  const { current, isActive, onActivate, onSelectSubtool } = useFamilySlot(engine, 'fill');
  const hotkeyDefinitions = useExtensionHotkeyDefinitions();
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const items: ToolFlyoutItem[] = useMemo(() => {
    const familyKeys = toolHotkeys(hotkeyDefinitions, customHotkeys, 'canvas.tool.gradient');
    const bucketKeys = toolHotkeys(hotkeyDefinitions, customHotkeys, 'canvas.tool.paintBucket');
    return [
      { hotkeys: familyKeys, icon: GradientIcon, id: 'gradient', label: t('widgets.canvas.tools.gradient') },
      {
        description: t('widgets.canvas.tools.paintBucketHelp'),
        // The bucket's own binding when it has one; otherwise the family key, which reaches it too.
        hotkeys: bucketKeys?.length ? bucketKeys : familyKeys,
        icon: PaintBucketIcon,
        id: 'paintBucket',
        label: t('widgets.canvas.tools.paintBucket'),
      },
    ];
  }, [customHotkeys, hotkeyDefinitions, t]);
  return (
    <ToolFamilyButton
      currentId={current}
      icon={FILL_FAMILY_ICONS[current]}
      isActive={isActive}
      items={items}
      label={t(`widgets.canvas.tools.${current}`)}
      onActivate={onActivate}
      onSelectSubtool={onSelectSubtool}
    />
  );
};

/**
 * Clears the center region's floating chrome, whose inset already includes the
 * gap below the islands. Outside the center the variable is undefined and the
 * strip falls back to its own top margin.
 */
const TOOL_STRIP_TOP = 'var(--wb-center-chrome-inset, var(--chakra-spacing-2))';

/**
 * Color picking stays a hold (Alt) or a key (I) rather than a strip button. Every tool stays available while a
 * generation runs or its results wait for review.
 */
const ToolStripRoot = ({ engine }: { engine: ToolStripEngine }) => {
  const { t } = useTranslation();
  const hotkeyDefinitions = useExtensionHotkeyDefinitions();
  const customHotkeys = useWorkbenchPreferenceSelector((preferences) => preferences.customHotkeys);
  const hotkeyFor = (hotkeyId: string) => toolHotkeys(hotkeyDefinitions, customHotkeys, hotkeyId)?.[0];

  return (
    <Box left="2" position="absolute" top={TOOL_STRIP_TOP} zIndex="2">
      <Toolbar aria-label={t('widgets.canvas.tools.label')}>
        <ToolStripButton
          engine={engine}
          hotkey={hotkeyFor('canvas.tool.view')}
          icon={HandIcon}
          label={t('widgets.canvas.tools.view')}
          toolId="view"
        />
        <ToolStripButton
          engine={engine}
          hotkey={hotkeyFor('canvas.tool.move')}
          icon={MousePointer2Icon}
          label={t('widgets.canvas.tools.move')}
          toolId="move"
        />
        <ToolStripButton
          engine={engine}
          hotkey={hotkeyFor('canvas.transformSelected')}
          icon={ScalingIcon}
          label={t('widgets.canvas.tools.transform')}
          toolId="transform"
        />
        <ToolStripButton
          description={t('widgets.canvas.tools.bboxHoldHelp')}
          engine={engine}
          // C, held or tapped, is the engine's own key for the Frame tool; a custom binding shows first when set.
          hotkey={hotkeyFor('canvas.tool.bbox') ?? 'c'}
          icon={ScanIcon}
          label={t('widgets.canvas.tools.bbox')}
          toolId="bbox"
        />
        <ToolStripButton
          engine={engine}
          hotkey={hotkeyFor('canvas.tool.brush')}
          icon={BrushIcon}
          label={t('widgets.canvas.tools.brush')}
          toolId="brush"
        />
        <ToolStripButton
          engine={engine}
          hotkey={hotkeyFor('canvas.tool.eraser')}
          icon={EraserIcon}
          label={t('widgets.canvas.tools.eraser')}
          toolId="eraser"
        />
        <ShapeFamilyButton engine={engine} hotkey={hotkeyFor('canvas.tool.shape')} />
        <FillFamilyButton engine={engine} />
        <ToolStripButton
          engine={engine}
          hotkey={hotkeyFor('canvas.tool.text')}
          icon={TypeIcon}
          label={t('widgets.canvas.tools.text')}
          toolId="text"
        />
        <SelectFamilyButton engine={engine} />
      </Toolbar>
    </Box>
  );
};

export const ToolStrip = Object.assign(ToolStripRoot, { Button: ToolStripButton });
