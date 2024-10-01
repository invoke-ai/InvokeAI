import { useFeatureStatus } from 'features/system/hooks/useFeatureStatus';
import { useMemo } from 'react';
import { type HotkeyCallback, type Options, useHotkeys } from 'react-hotkeys-hook';
import { useTranslation } from 'react-i18next';
import { assert } from 'tsafe';

type HotkeyCategory = 'app' | 'canvas' | 'viewer' | 'gallery' | 'workflows';

export type Hotkey = {
  id: string;
  category: string;
  title: string;
  desc: string;
  hotkeys: string[];
  platformKeys: string[][];
};

type HotkeyCategoryData = { title: string; hotkeys: Record<string, Hotkey> };

type HotkeysData = Record<HotkeyCategory, HotkeyCategoryData>;

const formatKeysForPlatform = (keys: string[], isMacOS: boolean): string[][] => {
  return keys.map((k) => {
    if (isMacOS) {
      return k.split('+').map((i) => i.replaceAll('mod', 'cmd').replaceAll('alt', 'option'));
    } else {
      return k.split('+').map((i) => i.replaceAll('mod', 'ctrl'));
    }
  });
};

export const useHotkeyData = (): HotkeysData => {
  const { t } = useTranslation();
  const isModelManagerEnabled = useFeatureStatus('modelManager');
  const isMacOS = useMemo(() => {
    return navigator.userAgent.toLowerCase().includes('mac');
  }, []);

  const hotkeysData = useMemo<HotkeysData>(() => {
    const data: HotkeysData = {
      app: {
        title: t('hotkeys.app.title'),
        hotkeys: {},
      },
      canvas: {
        title: t('hotkeys.canvas.title'),
        hotkeys: {},
      },
      viewer: {
        title: t('hotkeys.viewer.title'),
        hotkeys: {},
      },
      gallery: {
        title: t('hotkeys.gallery.title'),
        hotkeys: {},
      },
      workflows: {
        title: t('hotkeys.workflows.title'),
        hotkeys: {},
      },
    };

    const addHotkey = (category: HotkeyCategory, id: string, keys: string[]) => {
      data[category].hotkeys[id] = {
        id,
        category,
        title: t(`hotkeys.${category}.${id}.title`),
        desc: t(`hotkeys.${category}.${id}.desc`),
        hotkeys: keys,
        platformKeys: formatKeysForPlatform(keys, isMacOS),
      };
    };

    // App
    addHotkey('app', 'invoke', ['mod+enter']);
    addHotkey('app', 'invokeFront', ['mod+shift+enter']);
    addHotkey('app', 'cancelQueueItem', ['shift+x']);
    addHotkey('app', 'clearQueue', ['mod+shift+x']);
    addHotkey('app', 'selectCanvasTab', ['1']);
    addHotkey('app', 'selectUpscalingTab', ['2']);
    addHotkey('app', 'selectWorkflowsTab', ['3']);
    if (isModelManagerEnabled) {
      addHotkey('app', 'selectModelsTab', ['4']);
    }
    addHotkey('app', 'selectQueueTab', isModelManagerEnabled ? ['5'] : ['4']);
    addHotkey('app', 'focusPrompt', ['alt+a']);
    addHotkey('app', 'toggleLeftPanel', ['t', 'o']);
    addHotkey('app', 'toggleRightPanel', ['g']);
    addHotkey('app', 'resetPanelLayout', ['shift+r']);
    addHotkey('app', 'togglePanels', ['f']);

    // Canvas
    addHotkey('canvas', 'selectBrushTool', ['b']);
    addHotkey('canvas', 'selectBboxTool', ['c']);
    addHotkey('canvas', 'decrementToolWidth', ['[']);
    addHotkey('canvas', 'incrementToolWidth', [']']);
    addHotkey('canvas', 'selectEraserTool', ['e']);
    addHotkey('canvas', 'selectMoveTool', ['v']);
    addHotkey('canvas', 'selectRectTool', ['u']);
    addHotkey('canvas', 'selectViewTool', ['h']);
    addHotkey('canvas', 'selectColorPickerTool', ['i']);
    addHotkey('canvas', 'setFillToWhite', ['d']);
    addHotkey('canvas', 'fitLayersToCanvas', ['mod+0']);
    addHotkey('canvas', 'fitBboxToCanvas', ['mod+shift+0']);
    addHotkey('canvas', 'setZoomTo100Percent', ['mod+1']);
    addHotkey('canvas', 'setZoomTo200Percent', ['mod+2']);
    addHotkey('canvas', 'setZoomTo400Percent', ['mod+3']);
    addHotkey('canvas', 'setZoomTo800Percent', ['mod+4']);
    addHotkey('canvas', 'quickSwitch', ['q']);
    addHotkey('canvas', 'deleteSelected', ['delete', 'backspace']);
    addHotkey('canvas', 'resetSelected', ['shift+c']);
    addHotkey('canvas', 'transformSelected', ['shift+t']);
    addHotkey('canvas', 'filterSelected', ['shift+f']);
    addHotkey('canvas', 'undo', ['mod+z']);
    addHotkey('canvas', 'redo', ['mod+shift+z', 'mod+y']);
    addHotkey('canvas', 'nextEntity', ['alt+]']);
    addHotkey('canvas', 'prevEntity', ['alt+[']);
    addHotkey('canvas', 'applyFilter', ['enter']);
    addHotkey('canvas', 'cancelFilter', ['esc']);
    addHotkey('canvas', 'applyTransform', ['enter']);
    addHotkey('canvas', 'cancelTransform', ['esc']);

    // Workflows
    addHotkey('workflows', 'addNode', ['shift+a', 'space']);
    addHotkey('workflows', 'copySelection', ['mod+c']);
    addHotkey('workflows', 'pasteSelection', ['mod+v']);
    addHotkey('workflows', 'pasteSelectionWithEdges', ['mod+shift+v']);
    addHotkey('workflows', 'selectAll', ['mod+a']);
    addHotkey('workflows', 'deleteSelection', ['delete', 'backspace']);
    addHotkey('workflows', 'undo', ['mod+z']);
    addHotkey('workflows', 'redo', ['mod+shift+z', 'mod+y']);

    // Viewer
    addHotkey('viewer', 'toggleViewer', ['z']);
    addHotkey('viewer', 'swapImages', ['c']);
    addHotkey('viewer', 'nextComparisonMode', ['m']);
    addHotkey('viewer', 'loadWorkflow', ['w']);
    addHotkey('viewer', 'recallAll', ['a']);
    addHotkey('viewer', 'recallSeed', ['s']);
    addHotkey('viewer', 'recallPrompts', ['p']);
    addHotkey('viewer', 'remix', ['r']);
    addHotkey('viewer', 'useSize', ['d']);
    addHotkey('viewer', 'runPostprocessing', ['shift+u']);
    addHotkey('viewer', 'toggleMetadata', ['i']);

    // Gallery
    addHotkey('gallery', 'selectAllOnPage', ['mod+a']);
    addHotkey('gallery', 'clearSelection', ['esc']);
    addHotkey('gallery', 'galleryNavUp', ['up']);
    addHotkey('gallery', 'galleryNavRight', ['right']);
    addHotkey('gallery', 'galleryNavDown', ['down']);
    addHotkey('gallery', 'galleryNavLeft', ['left']);
    addHotkey('gallery', 'galleryNavUpAlt', ['alt+up']);
    addHotkey('gallery', 'galleryNavRightAlt', ['alt+right']);
    addHotkey('gallery', 'galleryNavDownAlt', ['alt+down']);
    addHotkey('gallery', 'galleryNavLeftAlt', ['alt+left']);
    addHotkey('gallery', 'deleteSelection', ['delete', 'backspace']);

    return data;
  }, [isMacOS, isModelManagerEnabled, t]);

  return hotkeysData;
};

type UseRegisteredHotkeysArg = {
  /**
   * The unique identifier for the hotkey. If `title` and `description` are omitted, the `id` will be used to look up
   * the translation strings for those fields:
   * - `hotkeys.${id}.label`
   * - `hotkeys.${id}.description`
   */
  id: string;
  /**
   * The category of the hotkey. This is used to group hotkeys in the hotkeys modal.
   */
  category: HotkeyCategory;
  /**
   * The callback to be invoked when the hotkey is triggered.
   */
  callback: HotkeyCallback;
  /**
   * The options for the hotkey. These are passed directly to `useHotkeys`.
   */
  options?: Options;
  /**
   * The dependencies for the hotkey. These are passed directly to `useHotkeys`.
   */
  dependencies?: readonly unknown[];
};

/**
 * A wrapper around `useHotkeys` that registers the hotkey with the hotkey registry.
 *
 * Registered hotkeys will be displayed in the hotkeys modal.
 */
export const useRegisteredHotkeys = ({ id, category, callback, options, dependencies }: UseRegisteredHotkeysArg) => {
  const hotkeysData = useHotkeyData();
  const keys = useMemo(() => {
    const _keys = hotkeysData[category].hotkeys[id]?.hotkeys;
    assert(_keys !== undefined, `Hotkey ${category}.${id} not found`);
    return _keys;
  }, [category, hotkeysData, id]);

  return useHotkeys(keys, callback, options, dependencies);
};
