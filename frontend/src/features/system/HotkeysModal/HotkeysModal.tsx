import {
  Accordion,
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
  Modal,
  ModalCloseButton,
  ModalContent,
  ModalOverlay,
  useDisclosure,
} from '@chakra-ui/react';
import React, { cloneElement, ReactElement } from 'react';
import HotkeysModalItem from './HotkeysModalItem';

type HotkeysModalProps = {
  /* The button to open the Settings Modal */
  children: ReactElement;
};

type HotkeyList = {
  title: string;
  desc: string;
  hotkey: string;
};

export default function HotkeysModal({ children }: HotkeysModalProps) {
  const {
    isOpen: isHotkeyModalOpen,
    onOpen: onHotkeysModalOpen,
    onClose: onHotkeysModalClose,
  } = useDisclosure();

  const appHotkeys = [
    { title: 'Invoke', desc: 'Generate an image', hotkey: 'Ctrl+Enter' },
    { title: 'Cancel', desc: 'Cancel image generation', hotkey: 'Shift+X' },
    {
      title: 'Focus Prompt',
      desc: 'Focus the prompt input area',
      hotkey: 'Alt+A',
    },
    {
      title: 'Toggle Options',
      desc: 'Open and close the options panel',
      hotkey: 'O',
    },
    {
      title: 'Pin Options',
      desc: 'Pin the options panel',
      hotkey: 'Shift+O',
    },
    {
      title: 'Toggle Gallery',
      desc: 'Open and close the gallery drawer',
      hotkey: 'G',
    },
    {
      title: 'Change Tabs',
      desc: 'Switch to another workspace',
      hotkey: '1-6',
    },
    {
      title: 'Theme Toggle',
      desc: 'Switch between dark and light modes',
      hotkey: 'Shift+D',
    },
    {
      title: 'Console Toggle',
      desc: 'Open and close console',
      hotkey: '`',
    },
  ];

  const generalHotkeys = [
    {
      title: 'Set Prompt',
      desc: 'Use the prompt of the current image',
      hotkey: 'P',
    },
    {
      title: 'Set Seed',
      desc: 'Use the seed of the current image',
      hotkey: 'S',
    },
    {
      title: 'Set Parameters',
      desc: 'Use all parameters of the current image',
      hotkey: 'A',
    },
    { title: 'Restore Faces', desc: 'Restore the current image', hotkey: 'R' },
    { title: 'Upscale', desc: 'Upscale the current image', hotkey: 'U' },
    {
      title: 'Show Info',
      desc: 'Show metadata info of the current image',
      hotkey: 'I',
    },
    {
      title: 'Send To Image To Image',
      desc: 'Send current image to Image to Image',
      hotkey: 'Shift+I',
    },
    { title: 'Delete Image', desc: 'Delete the current image', hotkey: 'Del' },
    { title: 'Close Panels', desc: 'Closes open panels', hotkey: 'Esc' },
  ];

  const galleryHotkeys = [
    {
      title: 'Previous Image',
      desc: 'Display the previous image in gallery',
      hotkey: 'Arrow left',
    },
    {
      title: 'Next Image',
      desc: 'Display the next image in gallery',
      hotkey: 'Arrow right',
    },
    {
      title: 'Toggle Gallery Pin',
      desc: 'Pins and unpins the gallery to the UI',
      hotkey: 'Shift+G',
    },
    {
      title: 'Increase Gallery Image Size',
      desc: 'Increases gallery thumbnails size',
      hotkey: 'Shift+Up',
    },
    {
      title: 'Decrease Gallery Image Size',
      desc: 'Decreases gallery thumbnails size',
      hotkey: 'Shift+Down',
    },
    {
      title: 'Reset Gallery Image Size',
      desc: 'Resets image gallery size',
      hotkey: 'Shift+R',
    },
  ];

  const inpaintingHotkeys = [
    {
      title: 'Select Brush',
      desc: 'Selects the inpainting brush',
      hotkey: 'B',
    },
    {
      title: 'Select Eraser',
      desc: 'Selects the inpainting eraser',
      hotkey: 'E',
    },
    {
      title: 'Quick Toggle Brush/Eraser',
      desc: 'Quick toggle between brush and eraser',
      hotkey: 'X',
    },
    {
      title: 'Decrease Brush Size',
      desc: 'Decreases the size of the inpainting brush/eraser',
      hotkey: '[',
    },
    {
      title: 'Increase Brush Size',
      desc: 'Increases the size of the inpainting brush/eraser',
      hotkey: ']',
    },
    {
      title: 'Hide Mask',
      desc: 'Hide and unhide mask',
      hotkey: 'H',
    },
    {
      title: 'Decrease Mask Opacity',
      desc: 'Decreases the opacity of the mask',
      hotkey: 'Shift+[',
    },
    {
      title: 'Increase Mask Opacity',
      desc: 'Increases the opacity of the mask',
      hotkey: 'Shift+]',
    },
    {
      title: 'Invert Mask',
      desc: 'Invert the mask preview',
      hotkey: 'Shift+M',
    },
    {
      title: 'Clear Mask',
      desc: 'Clear the entire mask',
      hotkey: 'Shift+C',
    },
    {
      title: 'Undo Stroke',
      desc: 'Undo a brush stroke',
      hotkey: 'Ctrl+Z',
    },
    {
      title: 'Redo Stroke',
      desc: 'Redo a brush stroke',
      hotkey: 'Ctrl+Shift+Z, Ctrl+Y',
    },
    {
      title: 'Lock Bounding Box',
      desc: 'Locks the bounding box',
      hotkey: 'Shift+Q',
    },
    {
      title: 'Quick Toggle Lock Bounding Box',
      desc: 'Hold to toggle locking the bounding box',
      hotkey: 'Q',
    },
    {
      title: 'Expand Inpainting Area',
      desc: 'Expand your inpainting work area',
      hotkey: 'Shift+J',
    },
  ];

  const renderHotkeyModalItems = (hotkeys: HotkeyList[]) => {
    const hotkeyModalItemsToRender: ReactElement[] = [];

    hotkeys.forEach((hotkey, i) => {
      hotkeyModalItemsToRender.push(
        <HotkeysModalItem
          key={i}
          title={hotkey.title}
          description={hotkey.desc}
          hotkey={hotkey.hotkey}
        />
      );
    });

    return (
      <div className="hotkey-modal-category">{hotkeyModalItemsToRender}</div>
    );
  };

  return (
    <>
      {cloneElement(children, {
        onClick: onHotkeysModalOpen,
      })}
      <Modal isOpen={isHotkeyModalOpen} onClose={onHotkeysModalClose}>
        <ModalOverlay />
        <ModalContent className="hotkeys-modal">
          <ModalCloseButton />

          <h1>Keyboard Shorcuts</h1>
          <div className="hotkeys-modal-items">
            <Accordion allowMultiple>
              <AccordionItem>
                <AccordionButton className="hotkeys-modal-button">
                  <h2>App Hotkeys</h2>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel>
                  {renderHotkeyModalItems(appHotkeys)}
                </AccordionPanel>
              </AccordionItem>

              <AccordionItem>
                <AccordionButton className="hotkeys-modal-button">
                  <h2>General Hotkeys</h2>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel>
                  {renderHotkeyModalItems(generalHotkeys)}
                </AccordionPanel>
              </AccordionItem>

              <AccordionItem>
                <AccordionButton className="hotkeys-modal-button">
                  <h2>Gallery Hotkeys</h2>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel>
                  {renderHotkeyModalItems(galleryHotkeys)}
                </AccordionPanel>
              </AccordionItem>

              <AccordionItem>
                <AccordionButton className="hotkeys-modal-button">
                  <h2>Inpainting Hotkeys</h2>
                  <AccordionIcon />
                </AccordionButton>
                <AccordionPanel>
                  {renderHotkeyModalItems(inpaintingHotkeys)}
                </AccordionPanel>
              </AccordionItem>
            </Accordion>
          </div>
        </ModalContent>
      </Modal>
    </>
  );
}
