import { useAppSelector } from 'app/store/storeHooks';
import { selectCustomHotkeys } from 'features/system/store/hotkeysSlice';
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
  defaultHotkeys: string[];
  platformKeys: string[][];
  isEnabled: boolean;
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
  const customHotkeys = useAppSelector(selectCustomHotkeys);
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

    const addHotkey = (category: HotkeyCategory, id: string, keys: string[], isEnabled: boolean = true) => {
      const hotkeyId = `${category}.${id}`;
      const effectiveKeys = customHotkeys[hotkeyId] ?? keys;
      data[category].hotkeys[id] = {
        id,
        category,
        title: t(`hotkeys.${category}.${id}.title`),
        desc: t(`hotkeys.${category}.${id}.desc`),
        hotkeys: effectiveKeys,
        defaultHotkeys: keys,
        platformKeys: formatKeysForPlatform(effectiveKeys, isMacOS),
        isEnabled,
      };
    };

    addHotkey('app', 'invoke', ['mod+enter']);
    addHotkey('app', 'invokeFront', ['mod+shift+enter']);
    addHotkey('app', 'cancelQueueItem', ['shift+x']);
    addHotkey('app', 'clearQueue', ['mod+shift+x']);
    addHotkey('app', 'selectGenerateTab', ['1']);
    addHotkey('app', 'selectCanvasTab', ['2']);
    addHotkey('app', 'selectUpscalingTab', ['3']);
    addHotkey('app', 'selectWorkflowsTab', ['4']);
    addHotkey('app', 'selectModelsTab', ['5']);
    addHotkey('app', 'selectQueueTab', ['6']);

    // Prompt/history navigation (when prompt textarea is focused)
    addHotkey('app', 'promptHistoryPrev', ['alt+arrowup']);
    addHotkey('app', 'promptHistoryNext', ['alt+arrowdown']);
    addHotkey('app', 'promptWeightUp', ['ctrl+arrowup']);
    addHotkey('app', 'promptWeightDown', ['ctrl+arrowdown']);

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
    addHotkey('canvas', 'setFillColorsToDefault', ['d']);
    addHotkey('canvas', 'toggleFillColor', ['x']);
    addHotkey('canvas', 'fitLayersToCanvas', ['mod+0']);
    addHotkey('canvas', 'fitBboxToCanvas', ['mod+shift+0']);
    addHotkey('canvas', 'fitBboxToLayers', ['shift+n']);
    addHotkey('canvas', 'setZoomTo100Percent', ['mod+1']);
    addHotkey('canvas', 'setZoomTo200Percent', ['mod+2']);
    addHotkey('canvas', 'setZoomTo400Percent', ['mod+3']);
    addHotkey('canvas', 'setZoomTo800Percent', ['mod+4']);
    addHotkey('canvas', 'quickSwitch', ['q']);
    addHotkey('canvas', 'deleteSelected', ['delete', 'backspace']);
    addHotkey('canvas', 'resetSelected', ['shift+c']);
    addHotkey('canvas', 'transformSelected', ['shift+t']);
    addHotkey('canvas', 'filterSelected', ['shift+f']);
    addHotkey('canvas', 'invertMask', ['shift+v']);
    addHotkey('canvas', 'undo', ['mod+z']);
    addHotkey('canvas', 'redo', ['mod+shift+z', 'mod+y']);
    addHotkey('canvas', 'nextEntity', ['alt+]']);
    addHotkey('canvas', 'prevEntity', ['alt+[']);
    addHotkey('canvas', 'applyFilter', ['enter']);
    addHotkey('canvas', 'cancelFilter', ['esc']);
    addHotkey('canvas', 'applyTransform', ['enter']);
    addHotkey('canvas', 'cancelTransform', ['esc']);
    addHotkey('canvas', 'applySegmentAnything', ['enter']);
    addHotkey('canvas', 'cancelSegmentAnything', ['esc']);
    addHotkey('canvas', 'toggleNonRasterLayers', ['shift+h']);
    addHotkey('canvas', 'fitBboxToMasks', ['shift+b']);
    addHotkey('canvas', 'toggleBbox', ['shift+o']);

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
    addHotkey('gallery', 'starImage', ['.']);

    return data;
  }, [customHotkeys, isMacOS, t]);

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
 * A wrapper around `useHotkeys` that adds a handler for a registered hotkey.
 */
export const useRegisteredHotkeys = ({ id, category, callback, options, dependencies }: UseRegisteredHotkeysArg) => {
  const hotkeysData = useHotkeyData();
  const data = useMemo(() => {
    const _data = hotkeysData[category].hotkeys[id];
    assert(_data !== undefined, `Hotkey ${category}.${id} not found`);
    return _data;
  }, [category, hotkeysData, id]);
  const _options = useMemo(() => {
    // If no options are provided, return the default. This includes if the hotkey is globally disabled.
    if (!options) {
      return {
        enabled: data.isEnabled,
      } satisfies Options;
    }
    // Otherwise, return the provided optiosn, but override the enabled state.
    return {
      ...options,
      enabled: data.isEnabled ? options.enabled : false,
    } satisfies Options;
  }, [data.isEnabled, options]);

  return useHotkeys(data.hotkeys, callback, _options, dependencies);
};
