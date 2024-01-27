import { useMemo } from 'react';
import { useTranslation } from 'react-i18next';

export type HotkeyListItem = {
  title: string;
  desc: string;
  hotkeys: string[][];
};

export type HotkeyGroup = {
  title: string;
  hotkeyListItems: HotkeyListItem[];
};

export const useHotkeyData = (): HotkeyGroup[] => {
  const { t } = useTranslation();

  const appHotkeys = useMemo<HotkeyGroup>(
    () => ({
      title: t('hotkeys.appHotkeys'),
      hotkeyListItems: [
        {
          title: t('hotkeys.invoke.title'),
          desc: t('hotkeys.invoke.desc'),
          hotkeys: [['Ctrl', 'Enter']],
        },
        {
          title: t('hotkeys.cancel.title'),
          desc: t('hotkeys.cancel.desc'),
          hotkeys: [['Shift', 'X']],
        },
        {
          title: t('hotkeys.cancelAndClear.title'),
          desc: t('hotkeys.cancelAndClear.desc'),
          hotkeys: [
            ['Shift', 'Ctrl', 'X'],
            ['Shift', 'Cmd', 'X'],
          ],
        },
        {
          title: t('hotkeys.focusPrompt.title'),
          desc: t('hotkeys.focusPrompt.desc'),
          hotkeys: [['Alt', 'A']],
        },
        {
          title: t('hotkeys.toggleOptions.title'),
          desc: t('hotkeys.toggleOptions.desc'),
          hotkeys: [['T'], ['O']],
        },
        {
          title: t('hotkeys.toggleGallery.title'),
          desc: t('hotkeys.toggleGallery.desc'),
          hotkeys: [['G']],
        },
        {
          title: t('hotkeys.toggleOptionsAndGallery.title'),
          desc: t('hotkeys.toggleOptionsAndGallery.desc'),
          hotkeys: [['F']],
        },
        {
          title: t('hotkeys.resetOptionsAndGallery.title'),
          desc: t('hotkeys.resetOptionsAndGallery.desc'),
          hotkeys: [['Shift', 'R']],
        },
        {
          title: t('hotkeys.maximizeWorkSpace.title'),
          desc: t('hotkeys.maximizeWorkSpace.desc'),
          hotkeys: [['F']],
        },
        {
          title: t('hotkeys.changeTabs.title'),
          desc: t('hotkeys.changeTabs.desc'),
          hotkeys: [['1 - 6']],
        },
      ],
    }),
    [t]
  );

  const generalHotkeys = useMemo<HotkeyGroup>(
    () => ({
      title: t('hotkeys.generalHotkeys'),
      hotkeyListItems: [
        {
          title: t('hotkeys.remixImage.title'),
          desc: t('hotkeys.remixImage.desc'),
          hotkeys: [['R']],
        },
        {
          title: t('hotkeys.setPrompt.title'),
          desc: t('hotkeys.setPrompt.desc'),
          hotkeys: [['P']],
        },
        {
          title: t('hotkeys.setSeed.title'),
          desc: t('hotkeys.setSeed.desc'),
          hotkeys: [['S']],
        },
        {
          title: t('hotkeys.setParameters.title'),
          desc: t('hotkeys.setParameters.desc'),
          hotkeys: [['A']],
        },
        {
          title: t('hotkeys.upscale.title'),
          desc: t('hotkeys.upscale.desc'),
          hotkeys: [['Shift', 'U']],
        },
        {
          title: t('hotkeys.showInfo.title'),
          desc: t('hotkeys.showInfo.desc'),
          hotkeys: [['I']],
        },
        {
          title: t('hotkeys.sendToImageToImage.title'),
          desc: t('hotkeys.sendToImageToImage.desc'),
          hotkeys: [['Shift', 'I']],
        },
        {
          title: t('hotkeys.deleteImage.title'),
          desc: t('hotkeys.deleteImage.desc'),
          hotkeys: [['Del']],
        },
      ],
    }),
    [t]
  );

  const galleryHotkeys = useMemo<HotkeyGroup>(
    () => ({
      title: t('hotkeys.galleryHotkeys'),
      hotkeyListItems: [
        {
          title: t('hotkeys.previousImage.title'),
          desc: t('hotkeys.previousImage.desc'),
          hotkeys: [['Arrow Left']],
        },
        {
          title: t('hotkeys.nextImage.title'),
          desc: t('hotkeys.nextImage.desc'),
          hotkeys: [['Arrow Right']],
        },
        {
          title: t('hotkeys.increaseGalleryThumbSize.title'),
          desc: t('hotkeys.increaseGalleryThumbSize.desc'),
          hotkeys: [['Shift', 'Up']],
        },
        {
          title: t('hotkeys.decreaseGalleryThumbSize.title'),
          desc: t('hotkeys.decreaseGalleryThumbSize.desc'),
          hotkeys: [['Shift', 'Down']],
        },
      ],
    }),
    [t]
  );

  const unifiedCanvasHotkeys = useMemo<HotkeyGroup>(
    () => ({
      title: t('hotkeys.unifiedCanvasHotkeys'),
      hotkeyListItems: [
        {
          title: t('hotkeys.selectBrush.title'),
          desc: t('hotkeys.selectBrush.desc'),
          hotkeys: [['B']],
        },
        {
          title: t('hotkeys.selectEraser.title'),
          desc: t('hotkeys.selectEraser.desc'),
          hotkeys: [['E']],
        },
        {
          title: t('hotkeys.decreaseBrushSize.title'),
          desc: t('hotkeys.decreaseBrushSize.desc'),
          hotkeys: [['[']],
        },
        {
          title: t('hotkeys.increaseBrushSize.title'),
          desc: t('hotkeys.increaseBrushSize.desc'),
          hotkeys: [[']']],
        },
        {
          title: t('hotkeys.decreaseBrushOpacity.title'),
          desc: t('hotkeys.decreaseBrushOpacity.desc'),
          hotkeys: [['Shift', '[']],
        },
        {
          title: t('hotkeys.increaseBrushOpacity.title'),
          desc: t('hotkeys.increaseBrushOpacity.desc'),
          hotkeys: [['Shift', ']']],
        },
        {
          title: t('hotkeys.moveTool.title'),
          desc: t('hotkeys.moveTool.desc'),
          hotkeys: [['V']],
        },
        {
          title: t('hotkeys.fillBoundingBox.title'),
          desc: t('hotkeys.fillBoundingBox.desc'),
          hotkeys: [['Shift', 'F']],
        },
        {
          title: t('hotkeys.eraseBoundingBox.title'),
          desc: t('hotkeys.eraseBoundingBox.desc'),
          hotkeys: [['Delete', 'Backspace']],
        },
        {
          title: t('hotkeys.colorPicker.title'),
          desc: t('hotkeys.colorPicker.desc'),
          hotkeys: [['C']],
        },
        {
          title: t('hotkeys.toggleSnap.title'),
          desc: t('hotkeys.toggleSnap.desc'),
          hotkeys: [['N']],
        },
        {
          title: t('hotkeys.quickToggleMove.title'),
          desc: t('hotkeys.quickToggleMove.desc'),
          hotkeys: [['Hold Space']],
        },
        {
          title: t('hotkeys.toggleLayer.title'),
          desc: t('hotkeys.toggleLayer.desc'),
          hotkeys: [['Q']],
        },
        {
          title: t('hotkeys.clearMask.title'),
          desc: t('hotkeys.clearMask.desc'),
          hotkeys: [['Shift', 'C']],
        },
        {
          title: t('hotkeys.hideMask.title'),
          desc: t('hotkeys.hideMask.desc'),
          hotkeys: [['H']],
        },
        {
          title: t('hotkeys.showHideBoundingBox.title'),
          desc: t('hotkeys.showHideBoundingBox.desc'),
          hotkeys: [['Shift', 'H']],
        },
        {
          title: t('hotkeys.mergeVisible.title'),
          desc: t('hotkeys.mergeVisible.desc'),
          hotkeys: [['Shift', 'M']],
        },
        {
          title: t('hotkeys.saveToGallery.title'),
          desc: t('hotkeys.saveToGallery.desc'),
          hotkeys: [['Shift', 'S']],
        },
        {
          title: t('hotkeys.copyToClipboard.title'),
          desc: t('hotkeys.copyToClipboard.desc'),
          hotkeys: [['Ctrl', 'C']],
        },
        {
          title: t('hotkeys.downloadImage.title'),
          desc: t('hotkeys.downloadImage.desc'),
          hotkeys: [['Shift', 'D']],
        },
        {
          title: t('hotkeys.undoStroke.title'),
          desc: t('hotkeys.undoStroke.desc'),
          hotkeys: [['Ctrl', 'Z']],
        },
        {
          title: t('hotkeys.redoStroke.title'),
          desc: t('hotkeys.redoStroke.desc'),
          hotkeys: [
            ['Ctrl', 'Shift', 'Z'],
            ['Ctrl', 'Y'],
          ],
        },
        {
          title: t('hotkeys.resetView.title'),
          desc: t('hotkeys.resetView.desc'),
          hotkeys: [['R']],
        },
        {
          title: t('hotkeys.previousStagingImage.title'),
          desc: t('hotkeys.previousStagingImage.desc'),
          hotkeys: [['Arrow Left']],
        },
        {
          title: t('hotkeys.nextStagingImage.title'),
          desc: t('hotkeys.nextStagingImage.desc'),
          hotkeys: [['Arrow Right']],
        },
        {
          title: t('hotkeys.acceptStagingImage.title'),
          desc: t('hotkeys.acceptStagingImage.desc'),
          hotkeys: [['Enter']],
        },
      ],
    }),
    [t]
  );

  const nodesHotkeys = useMemo<HotkeyGroup>(
    () => ({
      title: t('hotkeys.nodesHotkeys'),
      hotkeyListItems: [
        {
          title: t('hotkeys.addNodes.title'),
          desc: t('hotkeys.addNodes.desc'),
          hotkeys: [['Shift', 'A'], ['Space']],
        },
      ],
    }),
    [t]
  );

  const hotkeyGroups = useMemo<HotkeyGroup[]>(
    () => [appHotkeys, generalHotkeys, galleryHotkeys, unifiedCanvasHotkeys, nodesHotkeys],
    [appHotkeys, generalHotkeys, galleryHotkeys, unifiedCanvasHotkeys, nodesHotkeys]
  );

  return hotkeyGroups;
};
