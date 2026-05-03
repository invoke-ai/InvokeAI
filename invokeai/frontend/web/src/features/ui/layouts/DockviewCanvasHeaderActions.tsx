import { Flex, Kbd, Text } from '@invoke-ai/ui-library';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import type { IDockviewHeaderActionsProps } from 'dockview';
import { useCanvasManagerSafe } from 'features/controlLayers/contexts/CanvasManagerProviderGate';
import { selectLassoMode } from 'features/controlLayers/store/canvasSettingsSlice';
import { selectBbox } from 'features/controlLayers/store/selectors';
import type { Tool } from 'features/controlLayers/store/types';
import { IS_MAC_OS } from 'features/system/components/HotkeysModal/useHotkeyData';
import { atom } from 'nanostores';
import { Fragment, memo, useMemo } from 'react';
import { useTranslation } from 'react-i18next';

import { getCanvasToolModifierHints } from './canvasToolModifierHints';
import { WORKSPACE_PANEL_ID } from './shared';

const $fallbackTool = atom<Tool>('move');
const $fallbackToolBuffer = atom<Tool | null>(null);
const $fallbackTextSession = atom<null>(null);

type CanvasToolModifierHintKey = ReturnType<typeof getCanvasToolModifierHints>[number]['keys'][number];

const formatKey = (key: CanvasToolModifierHintKey, t: (key: string) => string) => {
  switch (key) {
    case 'mod':
      return IS_MAC_OS ? t('controlLayers.modifierHints.keys.command') : t('controlLayers.modifierHints.keys.control');
    case 'alt':
      return IS_MAC_OS ? t('controlLayers.modifierHints.keys.option') : t('controlLayers.modifierHints.keys.alt');
    case 'shift':
      return t('controlLayers.modifierHints.keys.shift');
    case 'space':
      return t('controlLayers.modifierHints.keys.space');
    case 'wheel':
      return t('controlLayers.modifierHints.keys.wheel');
    case 'arrows':
      return t('controlLayers.modifierHints.keys.arrows');
    case 'enter':
      return t('controlLayers.modifierHints.keys.enter');
    case 'esc':
      return t('controlLayers.modifierHints.keys.esc');
  }
};

export const DockviewCanvasHeaderActions = memo((props: IDockviewHeaderActionsProps) => {
  const { t } = useTranslation();
  const canvasManager = useCanvasManagerSafe();
  const lassoMode = useAppSelector(selectLassoMode);
  const bboxAspectRatioLocked = useAppSelector((state) => selectBbox(state).aspectRatio.isLocked);

  const tool = useStore(canvasManager?.tool.$tool ?? $fallbackTool);
  const toolBuffer = useStore(canvasManager?.tool.$toolBuffer ?? $fallbackToolBuffer);
  const textSession = useStore(canvasManager?.tool.tools.text.$session ?? $fallbackTextSession);

  const effectiveTool = useMemo<Tool>(() => {
    if (toolBuffer && (tool === 'view' || tool === 'colorPicker')) {
      return toolBuffer;
    }
    return tool;
  }, [tool, toolBuffer]);

  const hints = useMemo(() => {
    if (!canvasManager || props.activePanel?.id !== WORKSPACE_PANEL_ID) {
      return [];
    }

    return getCanvasToolModifierHints({
      tool: effectiveTool,
      lassoMode,
      bboxAspectRatioLocked,
      hasActiveTextSession: Boolean(textSession),
    });
  }, [bboxAspectRatioLocked, canvasManager, effectiveTool, lassoMode, props.activePanel?.id, textSession]);

  if (hints.length === 0) {
    return null;
  }

  return (
    <Flex
      h="full"
      alignItems="center"
      gap={4}
      pe={2}
      pointerEvents="none"
      userSelect="none"
      w="max-content"
      minW="max-content"
    >
      {hints.map((hint) => (
        <Flex key={hint.id} alignItems="center" gap={2} whiteSpace="nowrap">
          <Flex alignItems="center" gap={1} flexShrink={0}>
            {hint.keys.map((key, index) => (
              <Fragment key={`${hint.id}:${key}`}>
                {index > 0 && (
                  <Text fontSize="xs" color="base.500">
                    +
                  </Text>
                )}
                <Kbd fontSize="xs">{formatKey(key, t)}</Kbd>
              </Fragment>
            ))}
          </Flex>
          <Text fontSize="xs" color="base.300">
            {t(hint.labelKey)}
          </Text>
        </Flex>
      ))}
    </Flex>
  );
});

DockviewCanvasHeaderActions.displayName = 'DockviewCanvasHeaderActions';
